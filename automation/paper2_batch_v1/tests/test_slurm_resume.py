from pathlib import Path
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import common
import resume_frozen


class SlurmResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_finished_job_absent_from_active_queue_uses_accounting(self):
        with mock.patch.object(common,'execute',side_effect=['','123|FAILED|1:0\n']) as run:
            states,active=common.slurm_states(['123'])
        self.assertEqual(common.jobs_outcome(['123'],states,active),'FAILED')
        self.assertIn('-u',run.call_args_list[0].args[0])
        self.assertNotIn('-j',run.call_args_list[0].args[0])
        self.assertEqual(run.call_args_list[1].args[0][0],'sacct')

    def test_absence_from_both_queue_and_accounting_is_unknown(self):
        with mock.patch.object(common,'execute',side_effect=['','']):
            states,active=common.slurm_states(['123'])
        self.assertEqual(common.jobs_outcome(['123'],states,active),'UNKNOWN')

    def test_active_user_jobs_filtered_and_array_parent_kept(self):
        with mock.patch.object(common,'execute',side_effect=[
            '123_[2-4]|PENDING\n999|RUNNING\n','123_1|COMPLETED|0:0\n124|FAILED|1:0\n']):
            states,active=common.slurm_states(['123','124'])
        self.assertEqual(active,{'123'})
        self.assertEqual(common.jobs_outcome(['123','124'],states,active),'ACTIVE')

    def test_scheduler_error_is_not_treated_as_finished(self):
        error=subprocess.CalledProcessError(1,['squeue'],stderr='Scheduler unavailable')
        with mock.patch.object(common,'execute',side_effect=error) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                common.slurm_states(['123'])
        self.assertEqual(run.call_count,1)

    def test_accounting_error_is_not_treated_as_finished(self):
        error=subprocess.CalledProcessError(1,['sacct'],stderr='Accounting unavailable')
        with mock.patch.object(common,'execute',side_effect=['',error]):
            with self.assertRaises(subprocess.CalledProcessError):
                common.slurm_states(['123'])

    def test_command_failure_preserves_scheduler_stderr(self):
        error=subprocess.CalledProcessError(1,['squeue'],stderr='Scheduler unavailable')
        stream=io.StringIO()
        with mock.patch.object(common.subprocess,'run',side_effect=error),contextlib.redirect_stderr(stream):
            with self.assertRaises(subprocess.CalledProcessError):
                common.execute(['squeue'])
        self.assertIn('Scheduler unavailable',stream.getvalue())

    def frozen_run(self):
        run=self.root/'run'; tool=run/'tool'; (tool/'refs').mkdir(parents=True)
        for name,content in {
            'common.py':'old scheduler IO', 'controller.py':'frozen controller',
            'cohort.py':'frozen scientific processing', 'refs/model.py':'frozen model',
            'run.sh':'#!/bin/bash\ntrue\n'
        }.items():
            (tool/name).write_text(content)
        common.atomic_json(run/'tool_manifest.json',{
            str(p.relative_to(tool)):common.sha(p) for p in tool.rglob('*') if p.is_file()})
        common.atomic_json(run/'config.json',dict(account='test_account',partition='test_partition',qos='normal'))
        common.atomic_json(run/'controller_job.json',{'id':'123'})
        common.atomic_json(run/'state.json',{'systems':{'1abc_A':{'passed':True}}})
        return run

    def test_resume_separates_runtime_fix_and_keeps_frozen_science_and_results(self):
        run=self.frozen_run()
        before={name:(run/name).read_bytes() for name in ('tool/common.py','tool/cohort.py','tool_manifest.json','state.json')}
        with mock.patch.object(resume_frozen,'slurm_states',return_value=({'123':('FAILED','1:0')},set())),\
             mock.patch.object(resume_frozen,'execute',return_value='456\n') as submit:
            resume_frozen.launch(run)
        self.assertEqual(submit.call_args.args[0][0],'sbatch')
        for name,content in before.items(): self.assertEqual((run/name).read_bytes(),content)
        job=common.load_json(run/'controller_job.json'); self.assertEqual(job['id'],'456')
        manifest=Path(job['runtime_manifest']); evidence=common.load_json(manifest)
        self.assertFalse(evidence['scientific_code_changed'])
        runtime=manifest.parent/'tool'
        self.assertEqual((runtime/'cohort.py').read_bytes(),(run/'tool/cohort.py').read_bytes())
        self.assertEqual((runtime/'refs/model.py').read_bytes(),(run/'tool/refs/model.py').read_bytes())
        self.assertEqual(common.sha(runtime/'common.py'),common.sha(ROOT/'common.py'))
        compile((manifest.parent/'verify_and_run.py').read_text(),'checker','exec')
        subprocess.run(['bash','-n',run/'controller.sbatch'],check=True)

    def test_active_controller_does_not_submit_another(self):
        run=self.frozen_run()
        with mock.patch.object(resume_frozen,'slurm_states',return_value=({'123':('RUNNING','0:0')},{'123'})),\
             mock.patch.object(resume_frozen,'execute') as submit:
            resume_frozen.launch(run)
        submit.assert_not_called()
        self.assertFalse((run/'controller_runtimes').exists())

    def test_unknown_or_uncertain_controller_does_not_submit(self):
        run=self.frozen_run()
        with mock.patch.object(resume_frozen,'slurm_states',return_value=({},set())),\
             mock.patch.object(resume_frozen,'execute') as submit:
            with self.assertRaisesRegex(RuntimeError,'accounting unknown'):
                resume_frozen.launch(run)
        submit.assert_not_called()
        common.atomic_json(run/'controller_job.json',{'intent':True})
        with mock.patch.object(resume_frozen,'execute') as submit:
            with self.assertRaisesRegex(RuntimeError,'Uncertain controller submission'):
                resume_frozen.launch(run)
        submit.assert_not_called()

    def test_changed_runtime_blocks_new_submission(self):
        run=self.frozen_run()
        with mock.patch.object(resume_frozen,'slurm_states',return_value=({'123':('FAILED','1:0')},set())),\
             mock.patch.object(resume_frozen,'execute',return_value='456\n'):
            resume_frozen.launch(run)
        manifest=Path(common.load_json(run/'controller_job.json')['runtime_manifest'])
        (manifest.parent/'tool/cohort.py').write_text('changed scientific code')
        with mock.patch.object(resume_frozen,'slurm_states',return_value=({'456':('FAILED','1:0')},set())),\
             mock.patch.object(resume_frozen,'execute') as submit:
            with self.assertRaisesRegex(RuntimeError,'Changed Slurm runtime'):
                resume_frozen.launch(run)
        submit.assert_not_called()


if __name__=='__main__':
    unittest.main()
