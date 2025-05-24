def read_log_file(path):
    with open(path, 'r') as fl:
        epi_steps=[]
        returns=[]
        while True:
            line = fl.readline()
            if not line:
                break
            dc = eval(line)
            rw = dc['return']
            stp = dc['episode_steps']
            epi_steps.append(int(stp))
            returns.append(float(rw))
    return epi_steps, returns

def read_eval_log_file(path):
    with open(path, 'r') as fl:
        eval_steps=[]
        returns=[]
        while True:
            line = fl.readline()
            if not line:
                break
            dc = eval(line)
            rw = dc['return']
            stp = dc['eval_step']
            eval_steps.append(int(stp))
            returns.append(float(rw))
    return eval_steps, returns

def read_log_file_2(path):
    with open(path, 'r') as fl:
        epi_steps=[]
        returns=[]
        prv_stp = 0
        while True:
            line = fl.readline()
            if not line:
                break
            dc = eval(line)
            rw = dc['episode_reward']
            stp = dc['step']
            epi_steps.append(int(stp-prv_stp))
            returns.append(float(rw))
            prv_stp = stp
    return epi_steps, returns

def read_txt_file(path):
    with open(path, 'r') as fl:
        epi_steps = [int(float(step)) for step in fl.readline().split()]
        returns = [int(float(ret)) for ret in fl.readline().split()]
    return epi_steps, returns

def read_txt_file_2(path):
    returns = []
    epi_steps = []
    with open(path, 'r') as fl:
        lines = fl.readlines()
        for line in lines:
            spl = line.split()
            returns.append(float(spl[0]))
            epi_steps.append(int(spl[1]))
    return epi_steps, returns
