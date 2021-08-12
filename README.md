# <img src="img/deepracer.png?raw=true" height="70">

<p align = center>
  <img src="https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/-AWS-232F3E?logo=amazon-aws&logoColor=white"/>
  <img src="https://img.shields.io/badge/-Git-D51007?logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white"/>
</p>

# Deepracer2021
## About
This README describes my strategy for competing in the AWS Intern LIVE DeepRacer competition. This was an AWS Early Career Talent competition that gave interns the opportunity to create and train a DeepRacer model. Racers competed for the opportunity to win AWS DeepRacer merch. It was important to establish an action space, create a reward function for reinforcement learning, and experiment with various hyperparameters regulating the underlying 3-layer neural network during the model's development.



<p align="center">
<img src="img/finish_line.gif">
</p>


### About me
<p align="center">
<a href="https://www.linkedin.com/in/josh-hinton/"><img src="img/linkedin_profile_jh.png" width="45%"></a>
</p>

### Contents
- [About](#About)
- [Results](#Results)
- [Development](#Development)
  - [Initial Model](#Initial-Model)
  - [Qualifier Model](#Qualifier-Model)
  - [Finals Model](#Finals-Model)
- [Conclusion](#Conclusion)

## Results
### AWS Intern LIVE Finale 2021 (15th Place)
#### Track - Asia Pacific Bay Loop - Buildings
<p align="center">
<img src="img/final_results.png" width="38%">
</p>

### AWS Interns Qualifier 2021 (15th Place)
#### Track - Asia Pacific Bay Loop - Buildings
<p align="center">
<img src="img/qualifier_results.png" width="38%">
</p>

## Development
I'd like to preface this section by informing you that prior to the Deepracer event, my expertise of AI/ML was limited to a handful of YouTube videos. The end outcome of my experimentation was a set of 25 trained models and a newfound respect for AI/ML.

### Initial Model
I ran a simple Python reward function provided by AWS to familiarize myself with the concept of Deepracer models. I ran my model on Kuei Raceway for two hours; the reward graph is below.

<p align="center">
<img src="img/simple_tracks.png" width=46%>
</p>

The sub-rewards can be seen in this code snippet from [reward_simple.py](reward/dev/reward_simple.py):

```python
   # Give a very low reward by default
    reward = 1e-3

    # Give a high reward if no wheels go off the track and
    # the agent is somewhere in between the track borders
    if all_wheels_on_track and (0.5*track_width - distance_from_center) >= 0.05:
        reward = 1.0

```



### Qualifier Model
The organizers informed participants that all qualifiers and the Finals would be held on the Track - Asia Pacific Bay Loop - Building, which meant that I could begin training my models exclusively on that track.

#### Developing the reward function

### Finals Model
#### Redefining the action space

## Conclusion
