import torch
import numpy as np
import matplotlib.pyplot as plt

class NoiseSchedule():
    def __init__(self, T:int, schedule_type:str, B_0:float, B_T:float):
        self.T = T
        self.schedule_type = schedule_type
        self.B_0 = B_0
        self.B_T = B_T
        self.t = 0
        if self.schedule_type == 'linear':
            self.schedule = self.linear_schedule()
        elif self.schedule_type == 'cosine':
            self.schedule = self.cosine_schedule()
        elif self.schedule_type == 'cosine2':
            self.schedule = self.cosine_schedule2()
        

    def linear_schedule(self) -> callable:
        return lambda t: torch.tensor(self.B_0 + (self.B_T - self.B_0) * t / self.T)
    def cosine_schedule(self):
        return lambda t: torch.tensor(self.B_0 + 0.5 * (self.B_T - self.B_0) * (1 - np.cos(np.pi * (t + 0.5) / self.T)))

    def cosine_schedule2(self):
        s=0.008
        return lambda t: torch.tensor(np.cos(((t/self.T) + s)/(1+s) * np.pi/2)**2)
    
    
    def __getitem__(self, t:int) -> float:
        if t >= self.T:
            raise IndexError(f"t must be less than {self.T}")
        return self.schedule(t)
    
    def __next__(self):
        self.t += 1
        return self.schedule(self.t)
    def __iter__(self):
        return self
    
    def plot_schedule(self):
        x = range(0, self.T)
        y = [self.schedule(t) for t in x]
        plt.plot(x, y)
        plt.show()
    

class NoiseAdder():
    def __init__(self, noise_schedule:NoiseSchedule = None):            
        self.Schedule = noise_schedule if noise_schedule else NoiseSchedule(CFG.T, CFG.schedule_type, CFG.B_0, CFG.B_T)
    
    def add_noise(self, img:torch.Tensor, B_t):
        noise = torch.randn_like(img)
        x_t =  torch.sqrt(1 - B_t) * img + torch.sqrt(B_t) * noise
        return x_t
    
    def image_at_time_step(self, img:torch.Tensor, t):
        if t>=self.Schedule.T:
            raise(ValueError(f"The value should be lower than {self.Schedule.T}"))
        alpha_t_barre = torch.prod(torch.tensor([(1 - self.Schedule.schedule(s)) for s in range(t)]))
        noise = torch.randn_like(img)
        return torch.sqrt(alpha_t_barre) * img + torch.sqrt(1 - alpha_t_barre) * noise

    def image_at_time_step_cosine_v2(self, img:torch.Tensor, t):
        if t>=self.Schedule.T:
            raise(ValueError(f"The value should be lower than {self.Schedule.T}"))
        s=0.008
        # alpha_t_barre = torch.prod(torch.tensor([(1 - self.Schedule.schedule(s)) for s in range(t)]))
        f = lambda x: torch.tensor(np.cos(((x/self.Schedule.T) + s)/(1+s) * np.pi/2)**2)
        alpha_t_barre = f(t)/f(0)
        noise = torch.randn_like(img)
        return torch.sqrt(alpha_t_barre) * img + torch.sqrt(1 - alpha_t_barre) * noise


class CosineNoiseAdder():
    def __init__(self, T:torch.Tensor, s=0.008):
        self.T = T
        self.s = s # The value of s is selected such that sqrt(B_0) is slightly smaller than the pixel bin size 1/127.5, which gives s = 0.008. 
        
    def image_at_time_step(self, img:torch.Tensor, t):
        if t>=self.T:
            raise(ValueError(f"The value should be lower than {self.T}"))
        f = lambda x: torch.tensor(np.cos(((x/self.T) + self.s)/(1+self.s) * np.pi/2)**2)
        alpha_t_barre = f(t)/f(0)
        noise = torch.randn_like(img)
        noisy_image = torch.sqrt(alpha_t_barre) * img + torch.sqrt(1 - alpha_t_barre) * noise
        return noisy_image, noise