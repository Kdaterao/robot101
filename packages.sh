# quick script to set up 
# environmnet in one or two commands


#--- EARLY EXIT BEHAVIOUR ----

set -e # anything fails just exist early

#--- ENVIRONEMNT SET UP (THUNDERCOMPUTE SPECIFIC) ----


sudo apt update 

# ffmpeg (ubuntu specific)
sudo apt install -y ffmpeg

#tmux (to keep instance alive)
sudo apt install tmux


# huggin face (linux)
curl -LsSf https://hf.co/cli/install.sh | bash

# git submodules
git submodule update --init --recursive

# uv environments 
uv venv
source .venv/bin/activate
uv pip install -r "requirements.txt"

echo "Setup complete. Now login into huggingface!"