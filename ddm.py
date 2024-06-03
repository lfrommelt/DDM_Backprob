import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import OrderedDict
from math import ceil

class DDM:
  """
  A class representing a Drift Diffusion Model (DDM) for simulating decision-making processes.

  Attributes:
      z0 (float): Initial starting point.
      drift (float): Drift rate, representing the average rate of accumulation of evidence.
      noise (float): Standard deviation of the noise in the accumulation process.
      nrt (int): Non-decision time, representing the time required for non-decision processes.
      betas (list): List of coefficients for each dummy coded condition, respectively.
      dependent_variable (str): Indicates which parameter is dependent on conditions ('z0' or 'drift').
      conditions (list): List of condition names.
      t_upper (float): Upper threshold for decision boundary.
      t_lower (float): Lower threshold for decision boundary.
      max_timesteps (int): Maximum number of time steps for simulation.

  Methods:
      reset(self): Resets all logged data.
      get_noise(self): Returns noise sampled from a normal distribution.
      generate_trajectory(self, plausible=None, conditions=None): Simulates a trajectory according to the DDM.
      plot_last_trajectory(self): Plots the last generated trajectory.
      plot_histogram(self, conditions=[-1,0,1], normalize="absolute", title=None): Plots histograms of response times.
      compare(self, other, normalize="joint", title=None, conditions=[-1,1], names=("model", "other"), bincount=100): Visualizes histograms of two models for comparing.
  """
  def __init__(self, z0, drift, noise, nrt, betas, dependent_variable, conditions=["RainDropDown", "Neutral", "RainDropUp"], t_upper=1, t_lower=0, max_timesteps=1000):
    """
    Initializes the DDM with given parameters.

    Parameters:
        z0 (float): Initial starting point.
        drift (float): Drift rate, representing the average rate of accumulation of evidence.
        noise (float): Standard deviation of the noise in the accumulation process.
        nrt (int): Non-decision time, representing the time required for non-decision processes.
        betas (list): List of coefficients for each dummy coded condition, respectively.
        dependent_variable (str): Indicates which parameter is dependent on conditions ('z0' or 'drift').
        conditions (list): List of condition names.
        t_upper (float): Upper threshold for decision boundary.
        t_lower (float): Lower threshold for decision boundary.
        max_timesteps (int): Maximum number of time steps for simulation.
    """
    self.z0=z0
    self.drift=drift
    self.t_upper=t_upper
    self.t_lower=t_lower
    self.noise=noise
    self.nrt=nrt
    self.dependent_variable=dependent_variable
    self.max_timesteps=max_timesteps

    conditions=np.array(conditions)
    self.dummy_encodings=conditions[conditions!="Neutral"]
    if dependent_variable in ["z0", "drift"]:
      assert len(betas)==len(conditions)-1
    self.betas=np.array(betas)

    self.log=pd.DataFrame(columns=("plausible", "rt", "response", *self.dummy_encodings))
    self.trajectory=None

  def reset(self):
    '''
    Resets all logged data
    '''
    self.log=pd.DataFrame(columns=("plausible", "rt", "response", *self.dummy_encodings))
    self.trajectory=None

  def get_noise(self):
    """
    Returns noise sampled from a normal distribution.
    """
    return np.random.normal(0,self.noise)

  def generate_trajectory(self, plausible=None, conditions=None):
    """
    Simulates a trajectory according to the DDM. Either plausible or conditions has to be provided.

    Parameters:
        plausible (int): Plausibility of the condition (1=down is correct answer, -1=up is correct answer, 0=neutral stimulus was used).
        conditions (OrderedDict): Conditions dict with keys as condition names and values as condition values. Can also be pandas DataFrame.

    Returns:
        result Tuple(int, int): Result, consisting of reaction time (rt) and correctness (response).
    """
    assert not ((plausible is None) & (conditions is None)), "Please provide either plausible or conditions or both"

    # This is not general but hard-coded for our conventions, would require a mapping param in the constrctor
    if plausible is None:
      if conditions[self.dummy_encodings[0]]==1:
        plausible=1
      elif conditions[self.dummy_encodings[1]]==1:
        plausible=-1
      else:
        plausible=-1
    elif conditions is None:
      conditions=OrderedDict({condition: 0 for condition in self.dummy_encodings})
      if plausible==-1:
        conditions[self.dummy_encodings[1]]=1
      elif plausible==1:
        conditions[self.dummy_encodings[0]]=1


    t=self.nrt
    z=self.z0
    self.last_condition=plausible
    if self.dependent_variable=="z0":
      #z0 already is the offset, so we donot need an explicit offset
      z+=np.sum(self.betas*list(conditions.values()))
    self.trajectory=[z]
    while not z>=self.t_upper and not z<self.t_lower and not t>self.max_timesteps:
      t+=1
      z+=self.get_noise()
      if self.dependent_variable=="drift":
        # self.drift already is the offset of drift(plausibility), so we do not need an additional parameter for that
        z+=self.drift+np.sum(self.betas*list(conditions.values()))
      else:
        z+=self.drift
      self.trajectory.append(z)

    if z>=self.t_upper:
      response = 1
    elif z<=self.t_lower:
      response = -1
    else:
      response = 0

    logging_dict={"plausible":plausible, "rt":t, "response":response}
    logging_dict.update(conditions)
    self.log = self.log.append(logging_dict, ignore_index=True)

    return t, response

  def plot_last_trajectory(self):
    """
    Plots the last generated trajectory.
    """
    plt.plot((0,len(self.trajectory)+self.nrt),[self.t_upper, self.t_upper])
    plt.plot((0,len(self.trajectory)+self.nrt),[self.t_lower, self.t_lower])
    plt.plot(np.r_[[0], np.arange(len(self.trajectory))+self.nrt],np.r_[self.trajectory[0], self.trajectory])
    plt.title(f"condition: {self.last_condition}")
    plt.show()

  def plot_histogram(self, conditions=[-1,0,1], normalize="absolute", title=None):
    """
    Plots histograms of response times.

    Parameters:
        conditions (list): List of conditions to include in the histogram. Same values as plausible.
        normalize (str): Normalization method ('absolute', 'marginal', or 'joint').
        title (str): Title for the plot.
    """
    if normalize=="marginal":
      # up:=unplausible, pl:=plausible
      norm=np.zeros(len(conditions))
      for i, condition in enumerate(conditions):
        norm[i]=np.array([len(self.log[self.log["plausible"]==condition])])
    elif normalize=="joint":
      norm=np.array([len(self.log)])
      if conditions:
        norm=np.tile(norm, (len(conditions),1))
    else:
      norm=np.array([1])
      if conditions:
        norm=np.tile(norm, (len(conditions),1))


    if not conditions:
      if normalize=="marginal":
        raise ValueError("yea, how's that supposed to work? No conditions -> no normalizing over the respective marginal probs...")
      fig, axs = plt.subplots(1, 2)
      axs[0].title.set_text("correct")
      axs[0].set_ylabel("Relative Frequencies")
      axs[0].set_xlabel("Response Times")
      self.log[self.log["response"]==1]["rt"].hist(ax=axs[0], bins=100)
      #df_c.hist(ax=axs[0], label=f'correct', alpha=0.7, weights=(1/norm).repeat(len(df_pc)))
      axs[1].set_ylabel("Relative Frequencies")
      axs[1].set_xlabel("Response Times")
      self.log[self.log["response"]==-1]["rt"].hist(ax=axs[1], bins=100)
      axs[1].title.set_text("wrong")
    else:
      fig, axs = plt.subplots(1, 2)

      for i, condition in enumerate(conditions):

        df_pc=self.log[(self.log["response"]==1) & (self.log["plausible"]==condition)]["rt"]
        df_pc.hist(ax=axs[0], label=f'plausibility: {condition}', alpha=0.7, weights=(1/norm[i]).repeat(len(df_pc)))
        axs[0].title.set_text("correct")

        if normalize=="joint":
          axs[0].set_ylabel("Relative Frequencies")
        elif normalize=="marginal":
          axs[0].set_ylabel("Conditional relative Frequencies")
        else:
          axs[0].set_ylabel("Absolute Frequencies")

        axs[0].set_xlabel("Response Times")

        df_pw=self.log[(self.log["response"]==-1) & (self.log["plausible"]==condition)]["rt"]
        df_pw.hist(ax=axs[1], label=f'plausibility: {condition}', alpha=0.7, weights=(1/norm[i]).repeat(len(df_pw)))
        axs[1].title.set_text("wrong")

      plt.legend()
    if title:
      fig.suptitle(title)
    plt.show()


  def compare(self, other, normalize="joint", title=None, conditions=[-1,1], names=("model", "other"), bincount=100):
    """
    Compares histograms between two models.

    Parameters:
        other (DataFrame): DataFrame representing the data from the other model/distribution.
        normalize (str): Normalization method ('joint' or 'absolute'). No normalization by marginal distributions.
        title (str): Title for the plot.
        conditions (list): List of conditions to compare.
        names (tuple): Names of the models being compared.
        bincount (int): Number of bins for the histograms.
    """
    if normalize=="joint": # comparing after normalizing with marginal prob should not be done, the marginal distributions have to be learned as well
      norm_model=np.array([len(self.log)])
      norm_other=np.array([len(other)])
    else:
      norm_other=norm_model=np.array([1])

    fig, axs = plt.subplots(2, 2, figsize=(18,18))#len(conditions) instead of two

    min_self=self.log.groupby(["response", "plausible"]).min()
    max_self=self.log.groupby(["response", "plausible"]).max()

    min_other=other.groupby(["response", "plausible"]).min()
    max_other=other.groupby(["response", "plausible"]).max()

    mins=min_self.join(min_other, lsuffix='_self', rsuffix='_other').min(axis="columns").reset_index()
    max_=max_self.join(max_other, lsuffix='_self', rsuffix='_other').max(axis="columns").reset_index()

    for i, condition in enumerate(conditions):
      try:
        min_uc=mins[(mins["response"]==1)&(mins["plausible"]==condition)].iloc[0,2]
      except IndexError:
        min_uc=0
      try:
        max_uc=max_[(max_["response"]==1)&(max_["plausible"]==condition)].iloc[0,2]
      except:
        max_uc=self.max_timesteps
      range_uc=max_uc-min_uc
      step=ceil(range_uc/bincount)
      max_uc+=step
      bins_uc=np.arange(min_uc, max_uc, step)

      df_uc=self.log[(self.log["response"]==1) & (self.log["plausible"]==condition)]["rt"]
      df_uc.hist(ax=axs[i,0], label=names[0], alpha=0.7, weights=(1/norm_model).repeat(len(df_uc)),bins=bins_uc)

      df_uc_other=other[(other["response"]==1) & (other["plausible"]==condition)]["rt"]
      df_uc_other.hist(ax=axs[i,0], label=names[1], alpha=0.7, weights=(1/norm_other).repeat(len(df_uc_other)),bins=bins_uc)
      axs[i,0].title.set_text(f"correct, plausibility={condition}")

      if i==len(conditions)-1:

        if normalize=="joint":
          axs[i,0].set_ylabel("Relative Frequencies")
        else:
          axs[i,0].set_ylabel("Absolute Frequencies")

      try:
        min_uw=mins[(mins["response"]==-1)&(mins["plausible"]==condition)].iloc[0,2]
      except IndexError:
        min_uw=0
      try:
        max_uw=max_[(max_["response"]==-1)&(max_["plausible"]==condition)].iloc[0,2]
      except IndexError:
        max_uw=self.max_timesteps

      range_uw=max_uw-min_uw
      step=ceil(range_uw/bincount)
      max_uw+=step
      bins_uw=np.arange(min_uw, max_uw, step)

      df_uw=self.log[(self.log["response"]==-1) & (self.log["plausible"]==condition)]["rt"]
      df_uw.hist(ax=axs[i,1], label=names[0], alpha=0.7, weights=(1/norm_model).repeat(len(df_uw)),bins=bins_uw)

      df_uw_other=other[(other["response"]==-1) & (other["plausible"]==condition)]["rt"]
      df_uw_other.hist(ax=axs[i,1], label=names[1], alpha=0.7, weights=(1/norm_other).repeat(len(df_uw_other)),bins=bins_uw)
      axs[i,1].title.set_text(f"wrong, plausibility={condition}")

      axs[i,1].legend()
    if title:
      fig.suptitle(title)
    plt.show()

    
    
def generate_data_t(model, max_steps, z0=0.5, n=1000):
    # warning: changed for cum-distributions
    data=np.zeros((n,2))
    
    for i in range(n):
        z=z0
        
        run=True
        t=0
        while t<=max_steps:
            t+=1
            z = model.forward([z, np.random.random()-0.5])
            if z>=1:
                data[i]=[t,1]
                break
                
            elif z<=0:
                data[i]=[t,-1]
                break
        if t==max_steps:
            data[i]=[t, 1 if z>=0.5 else -1]
    return data


def plot_dist(train_model, max_steps):
    predictions=generate_data_t(train_model, max_steps, n=1000)
    df=pd.DataFrame()
    df["response"]=predictions[:,1]
    df["rt"]=predictions[:,0]
    ddm=DDM(0.5,0.2,0.1,10,(1,),"none", conditions=["s1, s2"])
    ddm.log=df
    #print(df)
    ddm.plot_histogram(conditions=[])
    
    
def generate_data_t_noiseless(model, max_steps, z0=0.5, n=1000, noise_replacement=0):
    data=np.zeros((n,2))
    
    for i in range(n):
        z=z0
        
        run=True
        t=0
        while t<=max_steps:
            t+=1
            z = model.forward([z, noise_replacement])#np.random.random()-0.5])
            if z>=1:
                data[i]=[t,1]
                break
                
            elif z<=0:
                data[i]=[t,-1]
                break

    return data

def plot_dist_noiseless(train_model, max_steps):
    predictions=generate_data_t_noiseless(train_model, max_steps, n=1000)
    df=pd.DataFrame()
    df["response"]=predictions[:,1]
    df["rt"]=predictions[:,0]
    ddm=DDM(0.5,0.2,0.1,10,(1,),"none", conditions=["s1, s2"])
    ddm.log=df
    #print(df)
    ddm.plot_histogram(conditions=[])
    
    
class Node:
    def __init__(self, free_params, constants, operation, derivative):
        self.free_params_dict=dict(zip(free_params,np.arange(len(free_params))))
        self.free_params=np.array(list(free_params.values()))
        self.constants=constants
        self.memory=[]
        self.operation=operation
        self.derivative=derivative
        self.gradients=[]
        
    def forward(self,x, v=False):
        self.memory.append(x)
        return self.operation(self.constants, self.free_params, x, v=v)
    
    def back(self, df, t=None, v=False):
        """
        params:
            df (float): gradient of the next operation aka derivative of all that happens after this node
        """
        x=self.memory[t]
        #x=self.memory.pop()
        gradients=self.derivative(self.constants, self.free_params, x, df)
        self.gradients.append(gradients[:len(self.free_params)])
        if v==True:
            print(f"gradient at {t}: {self.gradients[-1]}")
            print(f"for inputs x",x)
        return gradients
    
    def update(self, lr, v=False):
        """
        terrible efficiency, todo: use dict only for mapping index to name and work on an array
        """
        #for key in np.arange(len(self.free_params)):
        #print(self.gradients)
        if np.isnan(self.gradients).any():
            v=True
        self.free_params-=lr*np.sum(self.gradients, axis=0)
        if v:
            print("updates:",-lr*np.sum(self.gradients, axis=0))
            print("individual updates: ",-self.gradients)
            raise Exception
        self.gradients=[]
        self.memory=[]
    
def ddm_forward(constants, free_params, x, v=False):
    """
    expects x[0]=z and x[1]=noise
    """
    if v:
        print(f"{x[0]}+{x[1]}*{free_params[0]}+{free_params[1]}")
    return x[0]+x[1]*free_params[0]+free_params[1]

def ddm_backward(constants, free_params, x, df):
    """
    should give deriatives wrt noise, drift and z, nrt has to be somehow included in the loss function
    df (float): derivative of previous functions
    """
    #print(df, x)
    #return df*1, df*1, df*x[1] so far no z0
    return df*x[1], df*1, df*1#here

def l2_forward(constants, __, x):
    """
    x[0]=y, x[1]=true (convention: inputs first, constants last)
    """
    return 1/2*(x-constants)**2

def l2_backward(constants, __, x, ___):
    """
    gotta give gradient wrt input
    """
    return x-constants

def mean_l2_forward(constants, __, x):
    return 1/2*(constants - (1/2 * x[0] + 1/2 * x[1]))**2

def mean_l2_backward(constants, __, x, ___):
    df = -1/2*constants + 1/4*x[0] + 1/4*x[1]
    return df, df
"""
f = 1/2(1 - (1/2 x0 + 1/2 x1))**2
f = 1/2(1 - 1/2x0 - 1/2x1)**2

g(h) = 1/2h**2
h(x0, x1) = 1 - 1/2x0 - 1/2x1

df/dx = dh/dx (x) * dg/dx (h)

dh/dx0 = - 1/2

dg/dx = h

df/dx0 = - 1/2 * (1 - 1/2x0 - 1/2x1)
        = - 1/2 - 1/4 x0 - 1/4 x1

df/dx0 = 
"""