"""Independent synthetic audit probes, saved before implementation."""
import json
from pathlib import Path
import pytest
import torch
from self_healing_embodied_agents.agents import SelfHealingAgent
from self_healing_embodied_agents.env import TabletopManipulationEnv
from self_healing_embodied_agents.types import Action,ActionKind
from self_healing_embodied_agents.comparison import episode_metrics
from self_healing_embodied_agents.recovery import TaskOutcomeMemory
from self_healing_embodied_agents.training import TrainingMetrics,save_bundle,load_bundle
from self_healing_embodied_agents.world_model import ModelBundle,TransitionMLP
from scripts import run_comparison as runner


def valid_row():
    env=TabletopManipulationEnv(seed=4,perturbation='blocked_path')
    result=SelfHealingAgent(detector='failure',recovery='nominal').run_episode(env)
    return {**result.as_dict(),'event_log':result.event_log}

@pytest.mark.parametrize('damage',['success','bool-step','action','negative-step','counter-bool','counter-recovery','precision','task-bool','state-nan','late-alarm','false-trigger'])
def test_invalid_episode_cannot_enter_summary(damage):
    row=valid_row(); trans=[x for x in row['event_log'] if x['event']=='transition'];alarm=next(x for x in row['event_log'] if x['event']=='detection')
    if damage=='success':row['success']=0
    if damage=='bool-step':trans[0]['step']=True
    if damage=='action':trans[0]['action']='invented-action'
    if damage=='negative-step':alarm['step']=-1
    if damage=='counter-bool':row['true_failures']=True
    if damage=='counter-recovery':row['recovery_attempts']=99
    if damage=='precision':row['detection_precision']=.123
    if damage=='task-bool':trans[-1]['task_success']=1
    if damage=='state-nan':trans[0]['observed_state'][0]=float('nan')
    if damage=='late-alarm':alarm['step']=row['steps']+10
    if damage=='false-trigger':alarm['trigger']='residual_only';alarm['residual']=0.;alarm['threshold']=1.
    with pytest.raises(ValueError):episode_metrics(row)

@pytest.mark.parametrize('damage',['schema-bool','extra-key','training-extra','unknown-split','missing-field','duplicate-json'])
def test_invalid_study_config_rejected(tmp_path,damage):
    path=Path(__file__).resolve().parents[1]/'configs/comparison.json'
    config=json.loads(path.read_text())
    if damage=='schema-bool':config['schema_version']=True
    if damage=='extra-key':config['unknown']=1
    if damage=='training-extra':config['training']['unused']=1
    if damage=='unknown-split':config['evaluation']['unknown']=1
    if damage=='missing-field':config.pop('hypotheses')
    text=json.dumps(config)
    if damage=='duplicate-json':text=text.replace('"max_steps": 24','"max_steps": 24, "max_steps": 12')
    dest=tmp_path/'config.json';dest.write_text(text)
    with pytest.raises(ValueError):runner.load_study(dest)

def test_direct_runner_refuses_prior_artifact_before_training(tmp_path,monkeypatch):
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/comparison.json').read_text())
    (tmp_path/'summary.json').write_text('prior evidence')
    monkeypatch.setattr(runner,'train_transition_model',lambda **kw:pytest.fail('training started before collision rejection'))
    with pytest.raises((ValueError,FileExistsError)):runner.run_study(cfg,'development',tmp_path)
    assert (tmp_path/'summary.json').read_text()=='prior evidence'

def test_event_views_do_not_modify_environment_evidence():
    env=TabletopManipulationEnv(perturbation='transient_occlusion');env.reset()
    step=env.step(Action(ActionKind.MOVE_TO_OBJECT)); view=env.events;view[0]['failure']=False
    assert env.events[0]['failure'] is True
    step.events[0]['event']='modified'
    assert env.events[0]['event']=='transient_occlusion'

def test_task_memory_rejects_boolean_reward():
    memory=TaskOutcomeMemory(reward_sums={('f','c'):True},attempts={('f','c'):1})
    with pytest.raises(ValueError):memory.score('f','c')

@pytest.mark.parametrize('field,value',[('seed',True),('epochs',0),('train_samples',-1),('validation_mse',True),('train_episode_ids',[1,1])])
def test_invalid_training_metrics_do_not_publish(tmp_path,field,value):
    bundle=ModelBundle(TransitionMLP(),.1);metrics=TrainingMetrics(7,10,5,.01,.1)
    setattr(metrics,field,value)
    with pytest.raises(ValueError):save_bundle(bundle,metrics,tmp_path/'model.pt',tmp_path/'metrics.json')
    assert not list(tmp_path.iterdir())

def test_checkpoint_unknown_field_is_not_silently_ignored(tmp_path):
    bundle=ModelBundle(TransitionMLP(),.1);metrics=TrainingMetrics(7,10,5,.01,.1)
    path=tmp_path/'model.pt';save_bundle(bundle,metrics,path,tmp_path/'metrics.json')
    data=torch.load(path,weights_only=True);data['residual_threhsold']=.001;torch.save(data,path)
    with pytest.raises(ValueError):load_bundle(path)
