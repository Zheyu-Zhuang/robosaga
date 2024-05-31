import os
import yaml
import shutil
this_dir = os.path.dirname(os.path.abspath(__file__))
all_tasks = os.listdir(os.path.join(this_dir, "task"))
all_tasks = [os.path.basename(task) for task in all_tasks]
all_tasks = [task.split('.')[0] for task in all_tasks]  
#
aug_types = ['baseline', 'overlay', 'saga', 'soda']

f_base = 'train_diffusion_unet_hybrid'

for task in all_tasks:
    print(f'Processing task {task}' )
    for aug_type in aug_types:
        out_file = os.path.join(this_dir, f'train_diffusion_unet_hybrid_{task}_{aug_type}.yaml')
        if 'real' in task:
            f_path = os.path.join(this_dir,  f_base + f'_real_{aug_type}.yaml')
        else:
            f_path = os.path.join(this_dir,  f_base + f'_{aug_type}.yaml')
        shutil.copy(f_path, out_file)
        with open(out_file, 'r') as file:
            lines = file.readlines()

        new_lines = [f'  - task: {task}\n' if line.strip().startswith('- task:') else line for line in lines]
        if "lift" in task:
            new_lines = [f'  num_epochs: 300\n' if "num_epochs" in line else line for line in new_lines]

        with open(out_file, 'w') as file:
            file.writelines(new_lines)
