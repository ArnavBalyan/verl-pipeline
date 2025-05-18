# Agentica Project. 
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
# Standard library imports
from pprint import pprint

# Third party imports
import hydra
import ray
from omegaconf import OmegaConf

# Local application imports
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.ray_trainer_async import RayPPOAsyncTrainer
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.reward_manager import NaiveRewardManager
print("update")
from verl.trainer import SafeActorRolloutRefWorker


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo_pipeline(config)

def run_ppo_pipeline(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config, compute_score))

@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    print("Unified pool setup")
    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)
    from verl.trainer import GpuSupervisor

    GPU_SUPERVISORS = {
        i: GpuSupervisor.options(name=f"sup_{i}",
                                 lifetime="detached")
                    .remote(gpu_index=i)
        for i in range(config.trainer.n_gpus_per_node)
    }

    unified_pool_id = 'gpu_pool'
    total_gpus = config.trainer.n_gpus_per_node
    resource_pool_spec = {unified_pool_id: [total_gpus] * config.trainer.nnodes}
    mapping = {Role.Actor: unified_pool_id,
               Role.Rollout: unified_pool_id,
               Role.RefPolicy: unified_pool_id}
    resource_pool_manager = ResourcePoolManager(resource_pool_spec, mapping)

    reward_fn      = NaiveRewardManager(tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn  = NaiveRewardManager(tokenizer, num_examine=1, compute_score=compute_score)

    role_worker_mapping = {
        Role.Actor:     ray.remote(SafeActorRolloutRefWorker),
        Role.Rollout:   ray.remote(SafeActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(SafeActorRolloutRefWorker),
    }

    trainer = RayPPOAsyncTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=RayWorkerGroup,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
