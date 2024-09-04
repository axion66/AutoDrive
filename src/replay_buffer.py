import torch
import torchrl
from torchrl.data import PrioritizedReplayBuffer,ListStorage, LazyTensorStorage, TensorDictPrioritizedReplayBuffer
torch.manual_seed(0)





class PER:

    def __init__(self):
        self.buffer = PrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=ListStorage(10000))


    def sample(self,logger):
        sample,info = self.buffer.sample(batch_size=32,return_info=True)
        logger.info(f"Buffer sampled [sample]")
        return sample,info
    

    def add(self,data,logger):
        self.buffer.extend(data)
        logger.info(f"data added to the buffer [add]")

    def update_buffer(self,info,priority:torch.Tensor,logger):
        #priority = torch.ones(5) * 5
        #self.buffer.update_priority(info["index"], priority)

        self.buffer.update_priority(info['index'],priority)
        logger.info(f"Buffer updated [update_buffer]")


    def save_buffer(self,path,logger):  
        self.buffer.save(path)
        logger.info(f"PER saved at {path}. [save_buffer]")



    def load_buffer(self,path,logger):
        self.buffer.loads(path)
        logger.info(f"PER loaded from {path} [load_buffer]")



class PER_TensorDict:

    def __init__(self):
        self.buffer = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=1.1, storage=LazyTensorStorage(10000))


    def sample(self,logger):
        sample,info = self.buffer.sample(batch_size=32,return_info=True)
        logger.info(f"Buffer sampled [sample]")
        return sample,info
    

    def add(self,data,logger):
        #data = TensorDict({"a": torch.ones(10, 3), ("b", "c"): torch.zeros(10, 3, 1)}, [10])
        self.buffer.extend(data)
        logger.info(f"data added to the buffer [add]")

    def update_buffer(self,info,priority:torch.Tensor,logger):
        #priority = torch.ones(5) * 5
        #self.buffer.update_priority(info["index"], priority)
        self.buffer.set("td_error", priority)
        self.buffer.update_tensordict_priority(priority)
        logger.info(f"Buffer updated [update_buffer]")


    def save_buffer(self,path,logger):  
        self.buffer.save(path)
        logger.info(f"PER saved at {path}. [save_buffer]")



    def load_buffer(self,path,logger):
        self.buffer.loads(path)
        logger.info(f"PER loaded from {path} [load_buffer]")



    