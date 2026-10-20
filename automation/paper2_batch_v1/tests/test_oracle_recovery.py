import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import common
import recover_empty_oracle as recovery


def load_extractor(path):
    mda=types.ModuleType('MDAnalysis')
    analysis=types.ModuleType('MDAnalysis.analysis')
    analysis.rms=types.SimpleNamespace(rmsd=lambda a,b,**kw:float(np.sqrt(np.mean((a-b)**2))))
    spec=importlib.util.spec_from_file_location('test_extractor_'+path.parent.name,path)
    module=importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules,{'MDAnalysis':mda,'MDAnalysis.analysis':analysis}):
        spec.loader.exec_module(module)
    return module


class OracleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.original=ROOT.parents[1]/'src/pipeline/extract_atlas_core_rmsd.py'
        self.new=load_extractor(ROOT/'recovery/extract_atlas_core_rmsd.py')

    def tables(self):
        replica=pd.DataFrame([dict(system='2wpq_C',replica=1,quantile=.3,residue_index=i,
                                   prefix_core=True,full_core=True) for i in (1,2,3)])
        consensus=replica.drop(columns='replica').copy()
        return replica,consensus

    def test_empty_oracle_requires_opt_in_and_never_relaxes_prefix(self):
        r,c=self.tables(); c['full_core']=False
        with self.assertRaisesRegex(RuntimeError,'consensus_full_oracle'):
            self.new.selected_residue_indices(r,c,1,.3)
        selected=self.new.selected_residue_indices(r,c,1,.3,allow_empty_oracle=True)
        self.assertEqual(len(selected['consensus_full_oracle']),0)
        self.assertEqual(len(selected['replica_prefix']),3)
        for table,name in ((r,'replica_prefix'),(c,'consensus_prefix')):
            table['prefix_core']=False
            with self.assertRaisesRegex(RuntimeError,name):
                self.new.selected_residue_indices(r,c,1,.3,allow_empty_oracle=True)
            table['prefix_core']=True

    def simulate(self,module,empty=False):
        class Trajectory:
            def __len__(self): return 11
            def __getitem__(self,i): self.frame=i; return self
            def __iter__(self):
                for i in range(11):
                    self.frame=i; self.time=i*10000.; yield self
        tr=Trajectory()
        class Group:
            n_atoms=9
            @property
            def positions(self):
                return np.arange(27).reshape(9,3)+np.arange(9)[:,None]*tr.frame*.01
        universe=types.SimpleNamespace(trajectory=tr,select_atoms=lambda text:Group())
        sets={v:np.array([1,2,3]) for v in module.VARIANTS}
        if empty: sets['consensus_full_oracle']=np.array([],dtype=int)
        folder=self.root/(module.__name__+str(empty)); folder.mkdir(exist_ok=True)
        basic=pd.DataFrame(dict(time_ns=np.arange(11)*10.,rmsd_backbone_A=np.arange(11)*.1))
        with mock.patch.object(module,'create_core_group',return_value=Group()):
            out=module.process_replica_quantile('2wpq_C',universe,basic,sets,1,.3,folder,20.,80.,10000.,100.)
        return out,pd.read_csv(folder/'2wpq_C_R1_q30_core_rmsd.csv')

    def test_nonempty_series_match_executed_reference(self):
        original=load_extractor(self.original)
        old,old_csv=self.simulate(original)
        new,new_csv=self.simulate(self.new)
        pd.testing.assert_frame_equal(old_csv,new_csv)
        for first,second in zip(old[0],new[0]):
            for k,v in first.items():
                if k=='timeseries_csv': continue
                if isinstance(v,float):
                    np.testing.assert_allclose(v,second[k],equal_nan=True)
                else: self.assertEqual(v,second[k])

    def test_missing_oracle_is_nan_diagnostic_with_unchanged_finite_prefix(self):
        _,normal=self.simulate(self.new)
        report,missing=self.simulate(self.new,True)
        self.assertTrue(missing['consensus_full_oracle_rmsd_A'].isna().all())
        for variant in ('replica_prefix','consensus_prefix','replica_full_oracle'):
            np.testing.assert_array_equal(normal[variant+'_rmsd_A'],missing[variant+'_rmsd_A'])
            self.assertTrue(np.isfinite(missing[variant+'_rmsd_A']).all())
        row=next(r for r in report[0] if r['variant']=='consensus_full_oracle')
        self.assertEqual(row['n_core_residues'],0)
        self.assertEqual(row['diagnostic_status'],'UNAVAILABLE_EMPTY_ORACLE_CORE')
        self.assertTrue(np.isnan(row['mean_A']))
        affected=[r for r in report[1] if 'consensus_full_oracle' in r['comparison']]
        self.assertTrue(all(r['n_points']==0 and np.isnan(r['mae_A']) for r in affected))
        self.assertEqual(report[3]['diagnostic_status']['replica_prefix'],'AVAILABLE')

    def test_partial_nonfinite_comparison_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Partial/nonfinite'):
            self.new.comparison_metrics(np.array([0.,np.nan]),np.array([0.,1.]))

    def test_install_pins_reference_and_enables_only_core_stage(self):
        p=self.root/'project'; (p/'src/pipeline').mkdir(parents=True); (p/'slurm').mkdir()
        target=p/'src/pipeline/extract_atlas_core_rmsd.py'; shutil.copyfile(self.original,target)
        self.assertEqual(common.sha(target),recovery.REFERENCE_SHA)
        stage=p/'slurm/atlas_pipeline_stage.sbatch'
        stage.write_text('#!/bin/bash\npython "$PROJECT/src/pipeline/extract_atlas_core_rmsd.py" \\\n --system 2wpq_C\n')
        recovery.install_extractor(p)
        self.assertIn('--allow-empty-oracle',stage.read_text())
        recovery.install_extractor(p)
        self.assertEqual(stage.read_text().count('--allow-empty-oracle'),1)
        target.write_text('unknown source')
        with self.assertRaisesRegex(RuntimeError,'supported reference'):
            recovery.install_extractor(p)

    def test_recovery_cancels_only_owned_pending_jobs_on_failure(self):
        with mock.patch.object(recovery,'slurm_states',return_value=(
            {'100':('FAILED','1:0'),'101':('PENDING','0:0')},{'101'})),\
             mock.patch.object(recovery,'execute',side_effect=['101\n','']) as execute:
            with self.assertRaisesRegex(RuntimeError,'cohort state remains unchanged'):
                recovery.wait_recovery(['100','101'],20)
        self.assertEqual(execute.call_args_list[0].args[0],
            ['squeue','-h','-j','100,101','-t','PENDING','-o','%i'])
        self.assertEqual(execute.call_args_list[1].args[0],['scancel','101'])

    def test_success_adopts_only_verified_recovered_system_and_preserves_history(self):
        import controller
        run=self.root/'run'; run.mkdir(); work=self.root/'work'; (work/'maintenance').mkdir(parents=True)
        common.atomic_json(run/'config.json',dict(work=str(work),run_id='batch',poll=5))
        for name in ('tool_manifest','pipeline_manifest'):
            common.atomic_json(run/(name+'.json'),{})
        common.atomic_json(run/'selection.json',[dict(system='2wpq_C',dataset_group='new')])
        untouched={'passed':True,'ids':{'finalize':'88'},'release':'old_release'}
        failed={'failure':{'100':['FAILED','1:0']},'terminal':True,'ids':{'core_rmsd':'100'}}
        before=dict(systems={'1abc_A':untouched,'2wpq_C':failed})
        common.atomic_json(run/'state.json',before)
        repair=run/'recovery/2wpq_C_oracle_v1'; repair.mkdir(parents=True)
        common.atomic_json(repair/'tool_manifest.json',{})
        snapshot=repair/'pipeline_code'; (snapshot/'src/pipeline').mkdir(parents=True)
        (snapshot/'src/pipeline/extract_atlas_core_rmsd.py').write_text('reviewed')
        common.atomic_json(repair/'pipeline_manifest.json',{})
        output=work/'results/atlas_pipeline_runs/new/2wpq_C/batch_oracle_v1'
        core=output/'processed/new/2wpq_C/core_rmsd'; core.mkdir(parents=True)
        (output/'job_ids.txt').write_text('core_rmsd=200\nfinalize=201\n')
        common.atomic_json(core/'core_rmsd.json',{'trajectory_validation':[
            dict(replica=1,quantile=.3,diagnostic_status={
                'replica_prefix':'AVAILABLE','consensus_full_oracle':'UNAVAILABLE_EMPTY_ORACLE_CORE'})]})
        common.atomic_json(repair/'submission.json',{'ids':{'finalize':'201'}})
        with mock.patch.object(recovery,'slurm_states',return_value=({},set())),\
             mock.patch.object(recovery,'diagnosis',return_value={'original_failure':failed['failure']}),\
             mock.patch.object(recovery,'wait_recovery'),\
             mock.patch.object(controller,'verify_package',return_value='verified_new_release'),\
             mock.patch.object(recovery,'resume_original') as resume:
            recovery.perform(run,'2wpq_C')
        after=common.load_json(run/'state.json')['systems']
        self.assertEqual(after['1abc_A'],untouched)
        fixed=after['2wpq_C']; self.assertTrue(fixed['passed']); self.assertNotIn('failure',fixed)
        self.assertEqual(fixed['release'],'verified_new_release')
        self.assertEqual(fixed['recovery_history'][0]['previous']['failure'],failed['failure'])
        self.assertEqual(common.load_json(repair/'state_before_recovery.json'),before)
        resume.assert_called_once_with(run.resolve())

    def test_uncertain_controller_accounting_blocks_resume_without_submission(self):
        root=self.root/'run'; root.mkdir()
        common.atomic_json(root/'controller_job.json',{'id':'123'})
        with mock.patch.object(recovery,'slurm_states',return_value=({},set())),\
             mock.patch.object(recovery,'execute') as execute:
            with self.assertRaisesRegex(RuntimeError,'accounting unknown'):
                recovery.resume_original(root)
        execute.assert_not_called()


if __name__=='__main__':
    unittest.main()
