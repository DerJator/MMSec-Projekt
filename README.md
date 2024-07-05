# Usage of the NHR/RRZE HPC Cluster
## Introduction
Authors: Katharina Breininger, Jonas Utz

Please report errors or problems to katharina.breininger@fau.de

On a cluster you don't work directly with the computers performing your computations (the *compute nodes*). Instead, you connect to a special node (the *submit node*), submit your job there, and the cluster software will schedule it for execution on one of the compute nodes. As soon as the scheduler has found a node with the resources required for your job (and you haven't exceeded the maximum number of active jobs allowed for your account), the job is executed there. Keep this in mind while reading the remainder of this README.md.

**Some important information**:
- You can find basic information & documentation here:
https://doc.nhr.fau.de/getting_started/
https://doc.nhr.fau.de/access/overview/
- The HPC is only reachable within the university network (on campus or via VPN)
- Note that connection to the HPC works via SSH, more information below

:warning: If you have not worked with the HPC cluster before, please read these websites carefully, ask your colleagues and/or us in case of questions.

The HPC documentation is quite extensive, so this page aims to point you to the most relevant pages for the ADL challenge. *Please be invited to have a look at the rest of the documentation as well.* 


### Table of contents
- [Introduction](#introduction)
- [Account and logging in](#account-and-logging-in)
- [Working with the cluster and running jobs](#working-with-the-cluster-and-running-jobs)
  - [How to create an environment](#How-to-create-an-environment)
  - [Submitting jobs via slurm](#submitting-jobs-via-slurm)
  - [More in-depth information on slurm](#more-in-depth-information-on-slurm)
    - [Data handling](#data-handling)
    - [Useful commands for Slurm](#useful-commands-for-slurm)
    - [Debugging with an interactive slrum shell](#debugging-with-an-interactive-slurm-shell)
- [Tensorboard](#tensorboard)
- [Jupyter Notepooks](#jupyter)


## Account and logging in
You should have received an email with an invitation for an account from the NHR HPC. With this invitation, you can establish an account for the HPC and use the cluster for training your models.

You can log-in at the [HPC Portal](https://portal.hpc.fau.de/) to see the status of your account and edit your account.

Logging in to the server works via SSH. **Important**: In contrast to previous accounts, starting from now, you will only be able to log in **after you have uploaded an SSH-key**, i.e., it is no longer possible to log in with your account name and password.

Information on how to create an SSH key and how to upload it to the HPC portal here:
- https://doc.nhr.fau.de/access/ssh-command-line/
- https://doc.nhr.fau.de/hpc-portal/#upload-ssh-public-key-to-hpc-portal


After you have accomplished that, you can log in (via SSH, within the university network) to the dialog server: `cshpc.rrze.fau.de` 

You can do so via the command line, VS code or other options. More information here: https://doc.nhr.fau.de/access/ssh-command-line/

More general information on show SSH works can be found here: https://doc.nhr.fau.de/access/ssh-how-it-works/

Please check [Troubleshooting](
https://doc.nhr.fau.de/access/ssh-command-line/#troubleshooting)
first if you have any issues connecting before contacting us or the cluster admins. 

Now you are set to start working on the cluster.

## Working with the cluster and running jobs
Your main interaction (create python environment, upload code, potentially add additional data) will be with the login node.

**BUT**: To actually execute jobs (and use graphic cards) you need to submit a job on a compute node (not on the dialog server). 
For this, the HPC uses "slurm" (more information here: https://doc.nhr.fau.de/batch-processing/batch_system_slurm/).

>"When logging into an HPC system, you are placed on a login node. From there, you can manage your data, set up your workflow, and prepare and submit jobs. The login nodes are not suitable for computational work!"

Therefore, you will need to prepare a batch script to train your models.

We will provide you with some information here (or where to look for this information) on **how to create a python environment** and **how to submit jobs via slurm** next. We will include a brief note on data handling.

### How to create an environment:
On the cluster, the system-wide python installation is depricated; however, a large set of python modules is available via the module-system. Please have a look on how to use these modules at the documentation here:
- https://doc.nhr.fau.de/sdt/python/

You can also find information on how to create virtual environments / conda environments: 
- https://doc.nhr.fau.de/environment/python-env/
- https://doc.nhr.fau.de/environment/python-env/#conda-environments

Note that if you work together in a group, you can also share environments, i.e., not everyone has to install their own virtual environment.

### Submitting jobs via slurm:

**IMPORTANT**: Please don't overrequest graphics cards, and ideally start with RTX2080Ti (11GB) / RTX30803080 (10 GB) and let us know if you need larger cards. Please note that we cannot guarantee the availability of specific hardware.

As described before, you do not access the cluster nodes directly, but instead submit a job which than allocates a node for your job. The software to manage these jobs is called SLURM (https://slurm.schedmd.com). Generally you want to collect all commands and parameters that you need for your job in a single bash script which is then submitted to slurm. 

The command to submit jobs is called `sbatch`. To submit a job to the TinyGPU cluster use

```bash
sbatch.tinygpu [options] <job script>
```

After submission, sbatch will output the **Job ID** of your job. It can later be used for identification purposes and is also available as the environment variable `$SLURM_JOBID` in job scripts.

Below is an example for a job script which handles everything needed to train a neural network:

```bash
#!/bin/bash -l
#SBATCH --job-name=<your_job_name>
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail 
#SBATCH --time=00:15:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

source ~/.bashrc

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module purge
module load python
module load cuda
module load cudnn

# Conda
source activate <your_python_environment> # replace with the name of your conda env

# Copy data to `$TMPDIR` to have faster access, recommended esp. for long trainings
cp -r "$WORK/your-datasets" "$TMPDIR"
# in case you have to extract an archive, e.g. a dataset use:
# unzip "$WORK/your-dataset.zip" "$TMPDIR"
cd ${TMPDIR}

# create a temporary job dir on $WORK
mkdir ${WORK}/$SLURM_JOB_ID

# copy input file from location where job was submitted, and run 
cp -r ${SLURM_SUBMIT_DIR}/. .
mkdir -p output/logs/
mkdir -p output/checkpoints/

# Run training script (with data copied to node)
srun python training.py --dataset-path "$TMPDIR" --workdir "$TMPDIR" # add training parameters

# Create a directory on $HOME and copy the results from our training
mkdir ${HOME}/$SLURM_JOB_ID
cp -r ./output/. ${WORK}/$SLURM_JOB_ID

```

To submit this script type

```bash
sbatch.tinygpu TrainCluster.sh
```

from the directory where you have the code. 

### More in-depth information on slurm:

#### Data handling:
As mentioned, the dataset is provided in

```/home/janus/iwb6-datasets/FRAGMENTS``` 

as zip folders for the train and internal test data. We will provide additional external test data in due time.

**IMPORTANT**: Please avoid transferring a lot of small files, as this can slow down the cluster system heavily. Rather transfer zipped or tar-balled files and unpack them on the compute node as described in the example.

You can also follow the information here: https://doc.nhr.fau.de/data/datasets/


**Explanation of `sbatch` options**:

| Option                | Explanation                                                  |
| :-------------------- | :----------------------------------------------------------- |
| `--job-name=<name>`   | Specifies the name for the job. This is shown in the queue   |
| `--ntasks=<number>`   | number of parallel tasks. In our case always 1               |
| `--gres=gpu:1`        | Use GPU, if you want to use a specific GPU type use e.g. `--gres=gpu:rtx3080:1` |
| `--output=<filename>` | stdout is redirected into this file, i.e. when you call `print()` in python it will appear in this file |
| `--error=<filename`>  | stderr is redirected into this file. If your training is crashing check this file. |
| `--mail-type=<type>`  | Sends an email to you depending on `<type>`. Types are: `BEGIN`, `END`, `FAIL`, `TIME_LIMIT` or `ALL`. |
| `--time=HH:MM:SS`     | Specifies how long your job is running. If you exceed your time slurm will kill your job. (Max 24h). |
| `--partition=<name>` | Specifies which partition (and therefore which GPUs) to use, defaults to `work` on tinygpu (RTX2080Ti and RTX3080) |
|                       |                                                              |

#### Useful commands for Slurm:

| **Command**                         | Purpose                                                      |
| ----------------------------------- | ------------------------------------------------------------ |
| `squeue.tinygpu`                    | Displays information on jobs. Only the user’s own jobs are displayed. |
| `scancel.tinygpu <JobID>`           | Removes job from queue or terminates it if it’s already running. |
| `scontrol.tinygpu show job <JobID>` | Displays very detailed information on jobs.                  |

To avoid to type this commands every time you can define an alias by adding the following lines to  your `.bashrc` 

```bash
alias sq="squeue.tinygpu"
alias sb="sbatch.tinygpu"
```


#### Debugging with an interactive slurm shell

You can request an interactive shell, too. This is especially useful if you want to debug your code or check if everything works before submitting a hour-long training job. Please do not use the interactive shell for ongoing development.

To generate an interactive Slurm shell on one of the compute nodes, the following command has to be issued on the woody frontend:

```bash
salloc.tinygpu --gres=gpu:1 --time=00:30:00
```

When your job allocation is granted your connect automatically to the corresponding node. To use python within the interactive shell the following commands need to be issued: 

```bash
module load python
module load cuda
module load cudnn
source ~/.bashrc
source activate <your_python_environment> # replace with the name of your conda env
```

### Tensorboard

Tensorboard visualizes the progress of your training (e.g. the loss curve, accuracy, etc.). 

<img src="img/tebo.png" alt="tebo" style="zoom:33%;" />

You can either copy the logs to your local machine and run tensorboard from there or you can run tensorboard from the cluster frontend (`tinyx.nhr.fau.de`). 

1. Connect via VPN to the university network
2. `ssh` to `tinyx.nhr.fau.de`
3. run `tensorboard --logdir=/path/to/your/logs --bind_all`
4. Tensorboard will give you the URL to the tensorboard. Typically http://woody3.rrze.uni-erlangen.de:6006/ (Press CTRL+C to quit)
5. Open the URL in your browser

### Jupyter

If you prefer Jupyter Notebooks, find out how to work with them here:
https://doc.nhr.fau.de/apps/jupyter/


:exclamation: :exclamation: :exclamation: Have fun :exclamation: :exclamation: :exclamation: 