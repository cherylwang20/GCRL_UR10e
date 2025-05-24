import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd  
from read_files import read_log_file, read_log_file_2, read_eval_log_file
from read_files import read_txt_file, read_txt_file_2

print(pd.__version__)

def avg_graph(file_paths, plot_intervals, colors, labels, file_types, seeds, ylim, title, maxlen):
    sns.set_theme(rc={'figure.figsize':(10, 7)})
    sns.set_style("whitegrid")
    
    num_tasks = len(file_paths)

    for itr in range(num_tasks): 
        comb_data = []
        base_path = file_paths[itr] 
        
        print(labels[itr])

        for seed in seeds:
            eval_steps = None
            try:
                if file_types[itr] == 'log':
                    path = f'{base_path}/seed_{seed}/train.log'
                    epi_steps, returns = read_log_file(path)
                elif file_types[itr] == 'log2':
                    path = f'{base_path}/seed={seed}/train.log'
                    epi_steps, returns = read_log_file_2(path)
                elif file_types[itr] == 'eval':
                    path = f'{base_path}/seed_{seed}/eval.log'
                    eval_steps, returns = read_eval_log_file(path)
                elif file_types[itr] == 'txt':
                    path = f'{base_path}/seed_{seed}/return.txt'
                    epi_steps, returns = read_txt_file(path)
                elif file_types[itr] == 'txt2':
                    path = f'{base_path}/{seed}.txt'
                    epi_steps, returns = read_txt_file_2(path)
            except:
                print(f'\tSeed {seed}, error reading file.')
                continue

            if eval_steps: 
                end_step = plot_intervals[itr]
                rets = [] 
                for (i, step) in enumerate(eval_steps):
                    if step > end_step:
                        if len(rets) > 0:
                            comb_data.append([end_step, sum(rets)/len(rets), labels[itr]]) 
                            rets = []
                        end_step += plot_intervals[itr]

                    ret = returns[i]  
                    rets.append(ret) 
                    
                if len(rets) > 0:  
                    comb_data.append([end_step, sum(rets)/len(rets), labels[itr]]) 
            else:
                steps = 0
                end_step = plot_intervals[itr]
                rets = []
                
                for (i, epi_s) in enumerate(epi_steps): 
                    if steps + epi_s > end_step:
                        if len(rets) > 0:
                            comb_data.append([end_step, sum(rets)/len(rets), labels[itr]]) 
                            rets = []
                        end_step += plot_intervals[itr]

                    steps += epi_s
                    ret = returns[i]  
                    rets.append(ret)
                    
                if len(rets) > 0:  
                    comb_data.append([end_step, sum(rets)/len(rets), labels[itr]])
            
            if end_step != maxlen:
                print(f'\tSeed {seed} did not reach {maxlen}. Reached {end_step}.')
                
                
            
        df = pd.DataFrame(comb_data, columns=["Step", "Return", "Task"]) 
        ax1 = sns.lineplot(x="Step", y='Return', data=df,
                   color=sns.color_palette(colors)[itr], 
                   linewidth=2.0, label=labels[itr]) # , errorbar=None)
         
    
    # ax1.legend(loc='lower right')
    ax1.legend([], [], frameon=False)
    if ylim:
        plt.ylim(*ylim)
    plt.title(title)
    plt.savefig(f'imgs/{title}.png')
    plt.close()


if __name__ == "__main__": 
    file_paths = ["../results/HalfCheetah-v4_prop_sync", 
                  "../results/Hopper-v4_prop_sync", 
                  "../results/Ant-v4_prop_sync", 
                  "../results/Humanoid-v4_prop_sync",
                  "../results/Walker2d-v4_prop_sync",
                  "../results/cheetah_img_sync"]

    plot_intervals = [10000] * 6
    colors = 'bright'
    labels = ["HalfCheetah-v4", 
              "Hopper-v4", 
              "Ant-v4",
              "Humanoid-v4",
              "Walker2d-v4", 
              "cheetah_img"] 
    
    file_types =['eval'] * 6
    seeds = range(30)
    
    maxlens = [1000000] * 5 + [500000]
    ylim = None
    
    for i in range(len(labels)):
        title = labels[i]
        maxlen = maxlens[i]
        avg_graph(file_paths[i:i+1], plot_intervals[i:i+1], colors, labels[i:i+1], file_types[i:i+1], seeds, ylim, title, maxlen)