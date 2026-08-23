# A script to run our training jobs
# and then automatically shut off vm 
# for thundercompute 

# A script to run our training jobs
# and then automatically shut off vm 
# for thundercompute 

#--- DEFINE EARLY EXIT BEHAVIOR ----

cleanup() {
    echo "Stopping script (not shutting of instance)"  
    exit 1
}

trap cleanup INT

#--- GET INSTANCE ID ----

read -p "Enter Thunder instance ID: " INSTANCE_ID

#--- START TRAINING ----
source .venv/bin/activate
uv run src/train.py


#--- CREATE/REPLACE OUR SNAPSHOT -----
tnr snapshot create --instance-id "$INSTANCE_ID" --name training

#----- TURN OFF INSTANCE  ------
tnr delete "$INSTANCE_ID"

