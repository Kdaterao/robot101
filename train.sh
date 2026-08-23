# A script to run our training jobs
# and then automatically shut off vm 
# for thundercompute 

# A script to run our training jobs
# and then automatically shut off vm 
# for thundercompute 


# NOTE: WE DO NOT REALLY NEED A  SNAPSHOT JUST DOWNLOAD THE ENVIRONMENT IN LIKE 3 min its not that hard
#--- DEFINE EARLY EXIT BEHAVIOR ----

cleanup() {
    echo "Stopping script (not shutting off instance)"  
    exit 1
}

trap cleanup INT

# ffmpeg (ubuntu specific)
sudo apt update && sudo apt install -y ffmpeg

# uv environments 
uv venv
source .venv/bin/activate
uv pip install -r "requirements.txt"


#--- GET INSTANCE ID ----

read -p "Enter Thunder instance ID: " INSTANCE_ID

#--- START TRAINING ----
source .venv/bin/activate
uv run src/train.py


#----- TURN OFF INSTANCE  ------
tnr delete "$INSTANCE_ID"

