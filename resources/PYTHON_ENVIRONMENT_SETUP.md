# Python Virtual Environment Setup

This guide explains how to create, activate, and reuse a Python virtual environment on Linux.

In the examples below:

- `PyEnv` is the folder where the virtual environment will be stored.
- `clip` is the name of the virtual environment.
- Python 3.12 is used.

You may replace these names with names that are more appropriate for your project.

## 1. Create a Folder for Your Environments

Create a directory named `PyEnv` in your home directory:

```bash
mkdir -p ~/PyEnv
```

Navigate to the directory:

```bash
cd ~/PyEnv
```

## 2. Create a Virtual Environment

Create a virtual environment using Python 3.12:

```bash
python3.12 -m venv clip
```

This command creates a new folder named `clip` inside `~/PyEnv`.

The resulting directory structure will look like this:

```text
~/PyEnv/
└── clip/
```

## 3. Activate the Virtual Environment

Activate the environment with:

```bash
source ~/PyEnv/clip/bin/activate
```

After activation, the environment name should appear at the beginning of your terminal prompt:

```text
(clip) user@computer:~$
```

You can now install Python packages inside this environment. For example:

```bash
pip install numpy pandas
```

## 4. Verify the Active Environment

Check which Python executable is currently being used:

```bash
which python
```

The output should point to the Python executable inside your virtual environment:

```text
/home/your-username/PyEnv/clip/bin/python
```

You can also check the Python version:

```bash
python --version
```

## 5. Create a Convenient Activation Alias

To avoid typing the full activation command each time, you can add an alias to your `~/.bashrc` file.

Open the file using a text editor:

```bash
nano ~/.bashrc
```

Add the following line near the end of the file:

```bash
alias clip='source ~/PyEnv/clip/bin/activate'
```

Save the file and exit the editor.

For `nano`:

1. Press `Ctrl + O` to save.
2. Press `Enter` to confirm.
3. Press `Ctrl + X` to exit.

Apply the updated configuration without logging out:

```bash
source ~/.bashrc
```

You can now activate the environment by typing only:

```bash
clip
```

## 6. Deactivate the Environment

When you are finished working, deactivate the virtual environment with:

```bash
deactivate
```

## Complete Example

The following commands create and activate an environment named `tokenclip`:

```bash
mkdir -p ~/PyEnv
cd ~/PyEnv
python3.12 -m venv tokenclip
source ~/PyEnv/tokenclip/bin/activate
```

To create a convenient alias:

```bash
echo "alias tokenclip='source ~/PyEnv/tokenclip/bin/activate'" >> ~/.bashrc
source ~/.bashrc
```

After that, you can activate the environment at any time by running:

```bash
tokenclip
```

> **Note:** The alias will be available in new terminal sessions after it has been added to `~/.bashrc`.
