import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import common
import cohort
import controller
import cleanup as cleanup_module


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def zip(self,name='sample.zip',members=None):
        path=self.root/name
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
            for key,value in (members or {'sample.xtc':b'fixed trajectory bytes'}).items():
                z.writestr(key,value)
        return path

    def test_corrupted_archive_rejected(self):
        archive=self.zip()
        archive.write_bytes(archive.read_bytes().replace(b'fixed trajectory bytes',b'Fixed trajectory bytes'))
        with self.assertRaises(zipfile.BadZipFile):
            common.inspect_archive(archive)

    def test_traversal_rejected_before_unpack(self):
        archive=self.zip(members={'../escape':b'bad'})
        with self.assertRaisesRegex(RuntimeError,'Unsafe ZIP'):
            common.inspect_archive(archive)
        self.assertFalse((self.root.parent/'escape').exists())

    def test_zip_symlink_rejected(self):
        path=self.root/'link.zip'
        with zipfile.ZipFile(path,'w') as z:
            info=zipfile.ZipInfo('link'); info.create_system=3; info.external_attr=0o120777<<16
            z.writestr(info,'/tmp/elsewhere')
        with self.assertRaisesRegex(RuntimeError,'symlink'):
            common.inspect_archive(path)

    def test_duplicate_archive_members_rejected(self):
        archive=self.zip()
        with zipfile.ZipFile(archive,'a') as z:
            with self.assertWarns(UserWarning):
                z.writestr('sample.xtc',b'other')
        with self.assertRaisesRegex(RuntimeError,'Duplicate ZIP'):
            common.inspect_archive(archive)

    def test_unpack_resume_and_modified_input(self):
        archive=self.zip()
        verified=common.inspect_archive(archive)
        destination=self.root/'unpacked'
        common.unpack_archive(archive,destination,verified)
        before=common.stamp(destination/'sample.xtc')
        with mock.patch('common.shutil.disk_usage',return_value=mock.Mock(free=0)):
            common.unpack_archive(archive,destination,verified)
        self.assertEqual(before,common.stamp(destination/'sample.xtc'))
        (destination/'sample.xtc').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'differs'):
            common.unpack_archive(archive,destination,verified)

    def test_space_check_before_new_extraction(self):
        archive=self.zip(); verified=common.inspect_archive(archive)
        with mock.patch('common.shutil.disk_usage',return_value=mock.Mock(free=0)):
            with self.assertRaisesRegex(RuntimeError,'Insufficient'):
                common.unpack_archive(archive,self.root/'new',verified)
        self.assertFalse((self.root/'new').exists())

    def test_discovery_skips_known_excluded_partial_and_rejects_duplicate(self):
        sources=self.root/'sources.csv'
        common.write_csv(sources,[{'system':'1abc_A','release':'/old'}])
        base={'sources':str(sources),'acquired_exclusions':['3txs_C']}
        for group,system,kinds in [('g1','1abc_A',('analysis','protein')),
            ('g1','3txs_C',('analysis','protein')),('g1','2abc_A',('analysis','protein')),
            ('g1','4abc_A',('analysis',))]:
            directory=self.root/'raw/atlas'/group/system/'archives'; directory.mkdir(parents=True)
            for kind in kinds:
                (directory/f'{system}_{kind}.zip').write_bytes(b'dummy')
        result=controller.discover(self.root,base)
        self.assertEqual([r['system'] for r in result],['2abc_A'])
        directory=self.root/'raw/atlas/g2/2abc_A/archives'; directory.mkdir(parents=True)
        for kind in ('analysis','protein'):
            (directory/f'2abc_A_{kind}.zip').write_bytes(b'dummy')
        with self.assertRaisesRegex(RuntimeError,'two raw groups'):
            controller.discover(self.root,base)
        self.assertEqual(len(controller.discover(self.root,base,'g1')),1)

    def test_explicit_manifest_missing_pair_is_not_silently_skipped(self):
        (self.root/'raw/atlas/g1').mkdir(parents=True)
        sources=self.root/'sources.csv'; common.write_csv(sources,[dict(system='1abc_A',release='/old')])
        selection=self.root/'selection.csv'; common.write_csv(selection,[dict(system='2abc_A',dataset_group='g1')])
        with self.assertRaisesRegex(RuntimeError,'Missing/incomplete'):
            controller.discover(self.root,{'sources':str(sources)},manifest=selection)

    def test_group_bridge_cannot_leak_to_test(self):
        systems=['1abc_A','2abc_A','3abc_A']
        mapping,conflicts=cohort.groups_from_pairs(systems,[('1abc_A','2abc_A'),('2abc_A','3abc_A')],
              {'1abc_A':'train','2abc_A':'train','3abc_A':'test'})
        self.assertEqual(len(set(mapping.values())),1)
        self.assertEqual(conflicts,[systems])

    def test_same_pdb_grouping(self):
        mapping,conflicts=cohort.groups_from_pairs(['1abc_A','1abc_B'],[],{'1abc_A':'train','1abc_B':'train'})
        self.assertEqual(mapping['1abc_A'],mapping['1abc_B']); self.assertFalse(conflicts)

    def test_folds_keep_all_related_systems_and_are_order_invariant(self):
        rows=[dict(system_id=f'{i:04d}_A',group_id=f'g{i//2}',sample_id=f'{i}_R1',future_valid=True)
              for i in range(40)]
        a=cohort.make_folds(rows); b=cohort.make_folds(list(reversed(rows)))
        self.assertEqual(a,b)
        by_group={}
        for r in a:
            by_group.setdefault(r['group_id'],set()).add(r['outer_fold'])
        self.assertTrue(all(len(v)==1 for v in by_group.values()))
        self.assertEqual({r['outer_fold'] for r in a},set(range(5)))

    def test_not_enough_labeled_groups_blocks_training(self):
        with self.assertRaisesRegex(RuntimeError,'15 labeled groups'):
            cohort.make_folds([dict(system_id='1abc_A',group_id='g',future_valid=True)])

    def test_generalized_worker_has_no_fixed_cohort_counts(self):
        out=self.root/'worker.py'
        cohort.generalize_worker(ROOT/'refs/train_atlas222_grouped.py',out)
        tree=ast.parse(out.read_text())
        self.assertFalse(any(isinstance(n,ast.Constant) and type(n.value) is int and
                         n.value in (170,149,288,666,34) for n in ast.walk(tree)))
        compile(out.read_text(),str(out),'exec')
        self.assertIn("counts['fold_systems'][str(fold)]",out.read_text())

    def test_pipeline_snapshot_pins_paths_without_touching_original(self):
        project=self.root/'project'; (project/'src').mkdir(parents=True); (project/'slurm').mkdir()
        text='#!/bin/bash\nPROJECT="$HOME/old"\nWORK="$SCRATCH/old"\necho ok\n'
        for name in ('submit_atlas_system_pipeline.sh','atlas_pipeline_stage.sbatch'):
            (project/'slurm'/name).write_text(text)
        destination=self.root/'frozen'
        controller.snapshot_pipeline(project,destination,self.root,[dict(system='1abc_A',dataset_group='new')])
        self.assertEqual((project/'slurm/atlas_pipeline_stage.sbatch').read_text(),text)
        self.assertIn(str(destination),(destination/'slurm/atlas_pipeline_stage.sbatch').read_text())
        self.assertEqual(common.read_csv(destination/'manifests/atlas_systems.csv')[0]['status'],'staged')

    def test_accounting_lag_does_not_authorize_resubmission(self):
        self.assertEqual(common.jobs_outcome(['10'],{},set()),'UNKNOWN')
        self.assertEqual(common.jobs_outcome(['10'],{'10_1':('COMPLETED','0:0')},{'10'}),'ACTIVE')
        self.assertEqual(common.jobs_outcome(['10'],{'10_1':('FAILED','1:0')},set()),'FAILED')
        self.assertEqual(common.jobs_outcome(['10'],{'10_1':('COMPLETED','0:0'),'10_2':('COMPLETED','0:0')},set()),'PASS')
        self.assertEqual(common.jobs_outcome(['10'],{'10':('COMPLETED','1:0')},set()),'FAILED')

    def test_completed_stage_rejects_tampering(self):
        directory=self.root/'cohort'; directory.mkdir(); (directory/'data').write_text('original')
        controller.finish_step(directory)
        self.assertTrue(controller.completed_step(directory))
        (directory/'data').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'Changed completed'):
            controller.completed_step(directory)

    def test_fresh_attempt_keeps_failed_evidence(self):
        directory=self.root/'cohort'; directory.mkdir(); (directory/'failed.log').write_text('evidence')
        common.fresh_directory(directory)
        self.assertFalse((directory/'failed.log').exists())
        self.assertEqual(len(list((self.root/'attempts').rglob('failed.log'))),1)

    def test_pdb_chain_selection_and_first_model(self):
        path=self.root/'p.pdb'
        def atom(i,residue,chain,number):
            return f'ATOM  {i:5d}  CA  {residue:3s} {chain}{number:4d}    '+(' '*30)+'\n'
        path.write_text(atom(1,'ALA','A',1)+atom(2,'HID','A',2)+'ENDMDL\n'+atom(3,'GLY','A',3))
        self.assertEqual(cohort.pdb_sequence(path,'1abc_A'),'AH')
        self.assertEqual(cohort.pdb_sequence(path,'1abc_B'),'AH')
        path.write_text(atom(1,'ALA','A',1)+atom(2,'GLY','B',1))
        with self.assertRaisesRegex(RuntimeError,'Ambiguous'):
            cohort.pdb_sequence(path,'1abc_C')

    def test_future_window_reproduces_original_and_excludes_prefix(self):
        import numpy as np
        builder=cohort.load_function(ROOT/'refs/future_builder.py','reference_test')
        t=np.arange(10001)*.01; y=np.linspace(1,4,len(t))
        result=builder.recalculate(t,y,10.,float(y[1000:].mean()),lambda a,fast:1.)
        self.assertGreater(result['new_t0_ns'],20.)
        self.assertEqual(result['future_n_frames'],8000)
        self.assertTrue(result['future_valid'])
        with self.assertRaisesRegex(ValueError,'reproduce'):
            builder.recalculate(t,y,10.,999.,lambda a,fast:1.)

    def test_cleanup_full_verification_and_originals_retained(self):
        root=self.root/'run'; (root/'training/full_train').mkdir(parents=True)
        (root/'training/paired_audit.json').write_text('{}')
        (root/'training/full_train/metadata.json').write_text('{}')
        system='1abc_A'; raw=self.root/'raw'/system
        (raw/'archives').mkdir(parents=True)
        state={'systems':{system:{'passed':True}}}
        for kind in ('analysis','protein'):
            members={f'{system}_R{r}.xtc' if kind=='analysis' else f'{system}_prod_R{r}_fit.xtc':f'data{kind}{r}'.encode()
                     for r in (1,2,3)}
            members[system+'.pdb']=b'structure'
            path=raw/'archives'/f'{system}_{kind}.zip'
            with zipfile.ZipFile(path,'w') as z:
                for name,data in members.items():
                    z.writestr(name,data)
            state['systems'][system][kind]=common.inspect_archive(path)
            common.unpack_archive(path,raw/'unpacked'/f'{system}_{kind}',state['systems'][system][kind])
        with mock.patch('cleanup.execute',return_value=''),contextlib.redirect_stdout(io.StringIO()):
            result=cleanup_module.cleanup(root,[dict(system=system,raw=str(raw))],state)
            resumed=cleanup_module.cleanup(root,[dict(system=system,raw=str(raw))],state)
        self.assertEqual(result['deleted_this_attempt'],6)
        self.assertEqual(resumed,result)
        self.assertFalse(list((raw/'unpacked').rglob('*.xtc')))
        self.assertEqual(len(list((raw/'archives').glob('*.zip'))),2)
        self.assertEqual(len(list((raw/'unpacked').rglob('*.pdb'))),2)

    def controller_fixture(self):
        work=self.root/'work'; run=work/'results/paper2_automation/test_run'; run.mkdir(parents=True)
        project=self.root/'project'; (project/'src').mkdir(parents=True); (project/'slurm').mkdir()
        for name in ('submit_atlas_system_pipeline.sh','atlas_pipeline_stage.sbatch'):
            (project/'slurm'/name).write_text('#!/bin/bash\nPROJECT=old\nWORK=old\n')
        foundation=self.root/'foundation'; foundation.mkdir()
        source=foundation/'sources.csv'; common.write_csv(source,[dict(system='1abc_A',release='/old')])
        assigned=foundation/'assignment.csv'; common.write_csv(assigned,[dict(system_id='1abc_A',split='train',group_id='g1')])
        inventory=foundation/'inventory.csv'; common.write_csv(inventory,[dict(system='1abc_A')])
        fasta=foundation/'sequences.fasta'; fasta.write_text('>1abc_A\nAA\n')
        targets=foundation/'targets.csv'; common.write_csv(targets,[dict(sample_id='1abc_A_R1')])
        hashes=foundation/'source_hashes.json'; common.atomic_json(hashes,{})
        base=dict(sources=str(source),assignment=str(assigned),inventory=str(inventory),
                  sequences=str(fasta),future_rows=str(targets),source_hashes=str(hashes),acquired_exclusions=['3txs_C'])
        raw=work/'raw/atlas/new/2abc_A'; (raw/'archives').mkdir(parents=True)
        for kind in ('analysis','protein'):
            path=raw/'archives'/f'2abc_A_{kind}.zip'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('2abc_A.pdb',b'fixture')
        config=dict(work=str(work),run_id='test_run',project=str(project),group=None,manifest=None,
            base=None,parallel=2,poll=5,plan_only=False,cleanup_unpacked=False,
            account='a',partition='p',qos='q',blast_module='b')
        common.atomic_json(run/'config.json',config)
        (run/'tool').mkdir(); common.atomic_json(run/'tool_manifest.json',{})
        return run,work,base

    def controller_mocks(self,run,work,base,events):
        def execute(command,*args,**kwargs):
            command=list(map(str,command))
            if command[:2]==['bash','-n']:
                return ''
            if any('preflight_atlas_system.py' in p for p in command):
                events.append('preflight')
            elif any('submit_atlas_system_pipeline.sh' in p for p in command):
                if '--dry-run' in command:
                    events.append('dry_run')
                else:
                    events.append('submit')
                    output=work/'results/atlas_pipeline_runs/new/2abc_A/test_run'
                    output.mkdir(parents=True)
                    (output/'job_ids.txt').write_text('system=2abc_A\npreflight=100\nfinalize=101\n')
            return ''
        def prepare(config,base,sources,selection,out):
            events.append('prepare')
            for key,name in [('sources','sources.csv'),('assignment','system_assignment.csv'),
                ('inventory','source_inventory.csv'),('sequences','sequences.fasta'),
                ('future_rows','train_future_rmsd_rows.csv'),('source_hashes','source_hashes.json')]:
                (out/name).write_bytes(Path(base[key]).read_bytes())
            common.atomic_json(out/'summary.json',{'systems':2})
        def train(config,root):
            events.append('train'); return {'fixture':True}
        return [mock.patch('controller.resolve_base',return_value=base),
                mock.patch('controller.model_smoke_check'),
                mock.patch('controller.execute',side_effect=execute),
                mock.patch('controller.verify_package',return_value='/validated/prefix_sequences'),
                mock.patch('cohort.prepare',side_effect=prepare),
                mock.patch('controller.train',side_effect=train)]

    def test_controller_orders_preflight_dry_run_processing_and_training(self):
        run,work,base=self.controller_fixture(); events=[]
        with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            for patch in self.controller_mocks(run,work,base,events):
                stack.enter_context(patch)
            stack.enter_context(mock.patch('controller.slurm_states',return_value=(
                {'100':('COMPLETED','0:0'),'101':('COMPLETED','0:0')},set())))
            controller.controller(run)
        self.assertEqual(events,['preflight','dry_run','submit','prepare','train'])
        self.assertEqual(common.load_json(run/'summary.json')['status'],'COMPLETE')
        self.assertTrue((work/'results/paper2_automation/LATEST_SUCCESS.json').is_file())

    def test_resume_after_interruption_does_not_duplicate_submissions(self):
        run,work,base=self.controller_fixture(); events=[]
        with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            for patch in self.controller_mocks(run,work,base,events):
                stack.enter_context(patch)
            stack.enter_context(mock.patch('controller.slurm_states',return_value=({}, {'100','101'})))
            stack.enter_context(mock.patch('controller.time.sleep',side_effect=InterruptedError('simulated interruption')))
            with self.assertRaises(InterruptedError):
                controller.controller(run)
        self.assertEqual(events.count('submit'),1)
        with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            for patch in self.controller_mocks(run,work,base,events):
                stack.enter_context(patch)
            stack.enter_context(mock.patch('controller.slurm_states',return_value=(
                {'100':('COMPLETED','0:0'),'101':('COMPLETED','0:0')},set())))
            controller.controller(run)
        self.assertEqual(events.count('submit'),1)
        self.assertEqual(events.count('preflight'),1)
        self.assertEqual(events.count('dry_run'),1)
        self.assertEqual(events.count('train'),1)

    def test_plan_only_never_submits_processing_or_training(self):
        run,work,base=self.controller_fixture(); events=[]
        config=common.load_json(run/'config.json'); config['plan_only']=True
        common.atomic_json(run/'config.json',config)
        with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            for patch in self.controller_mocks(run,work,base,events):
                stack.enter_context(patch)
            controller.controller(run)
        self.assertEqual(events,['preflight','dry_run'])
        self.assertFalse((run/'summary.json').exists())

    def test_failure_blocks_training_and_is_not_marked_successful(self):
        run,work,base=self.controller_fixture(); events=[]
        with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            for patch in self.controller_mocks(run,work,base,events):
                stack.enter_context(patch)
            stack.enter_context(mock.patch('controller.slurm_states',return_value=(
                {'100':('FAILED','1:0'),'101':('CANCELLED','0:0')},set())))
            with self.assertRaisesRegex(RuntimeError,'blocked by failures'):
                controller.controller(run)
        self.assertNotIn('prepare',events); self.assertNotIn('train',events)
        self.assertFalse((run/'summary.json').exists())

    def test_uncertain_submission_is_not_reissued(self):
        run,work,base=self.controller_fixture(); events=[]
        config=common.load_json(run/'config.json'); config['plan_only']=True
        common.atomic_json(run/'config.json',config)
        with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            for patch in self.controller_mocks(run,work,base,events):
                stack.enter_context(patch)
            controller.controller(run)
        config['plan_only']=False; common.atomic_json(run/'config.json',config)
        state=common.load_json(run/'state.json'); state['systems']['2abc_A']['intent']=True
        common.atomic_json(run/'state.json',state)
        with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            for patch in self.controller_mocks(run,work,base,events):
                stack.enter_context(patch)
            with self.assertRaisesRegex(RuntimeError,'Uncertain submission'):
                controller.controller(run)
        self.assertNotIn('submit',events)

    def test_prepare_cohort_and_train_input_exclude_protected_labels(self):
        import types
        import numpy as np
        import pandas as pd
        root=self.root/'foundation'; root.mkdir()
        tool=ROOT
        systems=[f'{i:04d}_A' for i in range(1,32)]
        new='9999_A'
        alphabet='ARNDCQEGHILKMFPSTWYV'
        seq={s:alphabet[i//20]+alphabet[i%20]+'W' for i,s in enumerate(systems)}
        inv=[]; hashes={}; previous=[]
        cache=root/'cached.csv'; cache.write_text('cached bytes; must not be parsed as a new numeric series\n')
        for index,s in enumerate(systems):
            train=index<27
            master=root/f'{s}_master.csv'
            common.write_csv(master,[dict(system=s,replica=r,target_name='rmsd') for r in (1,2,3)])
            inv.append(dict(system=s,master_targets_csv=str(master),
                       **{f'basic_R{r}_csv':str(cache) for r in (1,2,3)}))
            if train:
                hashes[s]=dict(master_csv=str(master),master_sha256=common.sha(master),
                    replicas={str(r):dict(path=str(cache),sha256=common.sha(cache)) for r in (1,2,3)})
                for r in (1,2,3):
                    previous.append(dict(system_id=s,replica=r,sample_id=f'{s}_R{r}',group_id='grp_'+s,
                        origin='base',original_t0_ns=10.,original_valid=True,original_target_A=1.0,
                        original_mean_delta_A=0.,new_t0_ns=20.01,future_duration_ns=79.99,
                        future_n_frames=8000,future_g=1.,future_neff=8000.,future_valid=True,
                        future_target_A=1.1,common_valid=True))
        sources=root/'sources.csv'; common.write_csv(sources,[dict(system=s,release='/old/'+s) for s in systems])
        assignment=root/'assignment.csv'; common.write_csv(assignment,[dict(system_id=s,
            split='train' if i<27 else 'validation' if i<29 else 'test',group_id='grp_'+s)
            for i,s in enumerate(systems)])
        inventory=root/'inventory.csv'; common.write_csv(inventory,inv)
        fasta=root/'sequences.fasta'; fasta.write_text(''.join(f'>{s}\n{seq[s]}\n' for s in systems))
        targets=root/'targets.csv'; common.write_csv(targets,previous)
        source_hashes=root/'hashes.json'; common.atomic_json(source_hashes,hashes)
        base=dict(sources=sources,assignment=assignment,inventory=inventory,sequences=fasta,
                  future_rows=targets,source_hashes=source_hashes)
        release=self.root/'new_processed/prefix_sequences'; release.mkdir(parents=True)
        basic=release.parent/'basic'; basic.mkdir()
        master=release.parent/'master_targets'; master.mkdir()
        t=np.arange(10001)*.01; v=1+np.linspace(0,1,len(t))
        for r in (1,2,3):
            pd.DataFrame(dict(time_ns=t,rmsd_backbone_A=v)).to_csv(basic/f'{new}_R{r}_basic_observables.csv',index=False)
        common.write_csv(master/'master_targets_long.csv',[
            dict(system=new,replica=r,target_name='rmsd',t0_ns=10.,target_valid='true',
                 target_value=float(v[1000:].mean()),automatic_window_value=float(v[1000:].mean()))
            for r in (1,2,3)])
        raw=self.root/'raw'/new
        for kind in ('analysis','protein'):
            p=raw/'unpacked'/f'{new}_{kind}'/(new+'.pdb'); p.parent.mkdir(parents=True)
            p.write_text('ATOM      1  CA  TYR A   1                                  \n'
                         'ATOM      2  CA  TYR A   2                                  \n')
        output=self.root/'cohort'; output.mkdir()
        all_systems=systems+[new]
        features=['rmsd_backbone_A','radius_gyration_A','core_rmsd_q30_A','core_rmsd_q40_A',
                  'core_rmsd_q50_A','total_sasa_A2','helix_fraction','strand_fraction']
        def execute(command,*a,**kw):
            if '-m' in command:
                directory=output/'dataset'; directory.mkdir()
                np.savez_compressed(directory/'multisystem_dataset_raw.npz',
                    X_raw=np.ones((3*len(all_systems),201,8)),
                    system_ids=np.repeat(all_systems,3),
                    sample_ids=np.array([f'{s}_R{r}' for s in all_systems for r in (1,2,3)]),
                    time_ns=np.linspace(0,20,201),feature_names=features,
                    target_names=['rmsd','rg','sasa','helix','strand'],
                    target_values=np.full((3*len(all_systems),5),999999.))
            else:
                (output/'all_alignments.tsv').write_text(''.join(
                    f'{s}\t{s}\t100\t3\t3\t3\t1\t3\t1\t3\t0\t50\n' for s in all_systems))
            return ''
        fake=types.SimpleNamespace(timeseries=types.SimpleNamespace(statistical_inefficiency=lambda x,fast:1.))
        config=dict(run_id='new',pipeline_code=str(self.root),tool_code=str(tool),blast_module='fake')
        with mock.patch.dict(sys.modules,{'pymbar':fake}),mock.patch('cohort.execute',side_effect=execute):
            cohort.prepare(config,base,[dict(system=new,release=str(release))],
                           [dict(system=new,raw=str(raw))],output)
        summary=common.load_json(output/'summary.json')
        self.assertEqual(summary['systems'],32)
        self.assertEqual(summary['labeled_systems'],28)
        self.assertEqual(summary['labeled_replicas'],84)
        self.assertEqual(summary['base_future_targets_reused'],81)
        training=self.root/'training'; training.mkdir()
        cohort.training_bundle(config,output,training)
        with np.load(training/'dataset.npz',allow_pickle=False) as z:
            self.assertEqual(z['X_raw'].shape,(84,201,8))
            self.assertFalse(set(systems[27:])&set(z['system_ids']))
            self.assertTrue(np.isnan(z['target_values'][:,1:]).all())
            self.assertFalse((z['target_values']==999999.).any())
        self.assertEqual(common.load_json(training/'counts.json')['n_systems'],28)

    def test_audit_bootstrap_paired_groups_and_tampering(self):
        import numpy as np
        directory=self.root/'training'; directory.mkdir()
        systems=[f'{i:04d}_A' for i in range(1,23)]
        targets=[dict(system_id=s,sample_id=s+'_R1',group_id='g'+str(i//2),
                      future_valid=True,future_target_A=float(i)) for i,s in enumerate(systems)]
        # Use fifteen or more groups; only the first two systems share one group.
        for i,r in enumerate(targets):
            r['group_id']='g0' if i<2 else 'g'+str(i)
        folds=cohort.make_folds(targets)
        common.write_csv(directory/'outer_folds.csv',folds)
        common.write_csv(directory/'train_future_rmsd_rows.csv',targets)
        common.atomic_json(directory/'counts.json',dict(n_systems=22,n_groups=21,n_replicas=22))
        mapping={r['system_id']:r for r in folds}
        result=[]
        errors=dict(zip(cohort.MODELS,[.3,.2,.4,.15,.1]))
        for f in range(5):
            path=directory/f'fold_{f}'; path.mkdir()
            rr=[]
            for t in targets:
                if mapping[t['system_id']]['outer_fold']==f:
                    for model in cohort.MODELS:
                        common_fields=dict(fold=f,system_id=t['system_id'],group_id=t['group_id'],
                                           model=model,true=t['future_target_A'],predicted=t['future_target_A']+errors[model])
                        rr.append(dict(sample_id=t['sample_id'],**common_fields))
                        result.append(dict(n_valid_replicas=1,**common_fields))
            common.write_csv(path/'replica_predictions.csv',rr)
            ev={r['group_id'] for r in folds if r['outer_fold']==f}
            fit={r['group_id'] for r in folds}-ev
            common.atomic_json(path/'metadata.json',dict(outer_eval_groups=sorted(ev),
                ridge_inner_eval_groups=[sorted(fit)],
                selection={'gru_residual_rmsd_pilot':dict(inner_eval_groups=sorted(fit))}))
        common.write_csv(directory/'system_predictions.csv',result)
        report=cohort.audit_training(directory)
        contrast=report['comparisons'][0]
        self.assertAlmostEqual(contrast['delta_mae'],-.05)
        np.testing.assert_allclose(contrast['bootstrap_95'],[-.05,-.05],atol=1e-10)
        broken=common.read_csv(directory/'fold_0/replica_predictions.csv')
        broken[0]['true']='9999'
        common.write_csv(directory/'fold_0/replica_predictions.csv',broken)
        with self.assertRaisesRegex(RuntimeError,'truth differs'):
            cohort.audit_training(directory)


if __name__=='__main__':
    unittest.main()
