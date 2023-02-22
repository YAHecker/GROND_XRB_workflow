#!/bin/sh

python AVmod1.py &
python AVmod2.py &
python AVmod3.py &
python AVmod4.py &
python AVmod5.py &

wait
echo Done with all models for A0620.
