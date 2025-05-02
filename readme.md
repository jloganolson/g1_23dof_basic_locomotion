This is my first working(ish) attempt at making and deploygin a balancing policy for the 23DOF G1.

![policy output](output.gif)

It borrows heavily from the different unitree repos and mujoco playground examples.
*If you make any improvements or create a separate balance policy, please let me know - I want to learn!*

### Qualifiers
* It's brittle and jank but it kind of works (i think?), so I thought I'd put it as a standalone repo on github in case this helps out someone else.
* I didn't cleanup/comment/organize the code at all. It's pretty small but might be difficult to follow, - you are absolutely welcome to ask me questions via email/twitter.
* This code is copy/pasted from the [main repo I was derping around in](https://github.com/jloganolson/g1_mjx_helloworld) and I didn't test it thoroughly - if you run into issues, again, you're totally welcome to ping me.



### Instructions
Setup/pip install the following repos 
(https://github.com/unitreerobotics/unitree_sdk2_python)
(https://github.com/google-deepmind/mujoco_playground)
(https://github.com/google/brax)

(note: doing this from memory so there may be more...)

From inside the script folder, run `python main.py` to train a policy
To deploy that policy run `python deploy_real.py` -- be sure to replace `CHECKPOINT_PATH` with your own and have an environment variable for `NETWORK_CARD_NAME` (per their docs)



