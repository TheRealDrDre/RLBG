import os
from psychopy import visual, event, core
from psychopy.visual import ShapeStim

os.chdir('C:\\Users\\ausma_000\\Documents\\gp\\RLBG\\psypyPSS\\PSS')

import PSSTask as pt

task = pt.PSS_Task()

trials = [element for tupl in task.TRAINING_BLOCK for element in tupl]


#create a visual window
win = visual.Window([400,400],winType='pyglet')

vert = visual.Line(win,start = (0,-0.1),end = (0,0.1),lineWidth=2)
horz = visual.Line(win,start = (-0.1,0),end = (0.1,0),lineWidth=2)

stimulus = visual.TextStim(win, text = '')
prompt1 = visual.TextStim(win, text = 'Take it!', pos = (-.5, 0))
prompt2 = visual.TextStim(win, text = 'Leave it!', pos = (.5, 0))
rew = visual.TextStim(win, text = '')

keys = list()
stim = list()
rewards = list()
fixOnset = list()
stimOnset = list()
promptOnset = list()
rewOnset = list()

exptTime = core.Clock()

for t in trials:
    vert.draw()
    horz.draw()
    win.callOnFlip(fixOnset.append,exptTime.getTime())
    win.flip()
    core.wait(1.0)
    
    stim.append(t)
    stimulus.text = t
    stimulus.draw()
    win.callOnFlip(stimOnset.append,exptTime.getTime())
    win.flip()
    core.wait(2.0)
    
    prompt1.draw()
    prompt2.draw()
    win.callOnFlip(promptOnset.append,exptTime.getTime())
    win.flip()
    
    count = 0
    k = ('','')
    while count < 1:
        k = event.waitKeys(maxWait = 2.0, keyList = ['left','right'], timeStamped=exptTime)
        print(k)
        count =+ 1
    
    if k is None:
        k = [('',''),('','')]
    
    
    if k[0][0]=='left':
        value = task.get_reward(t)
        if value==1:
            reward = 'Correct!'
        elif value==-1:
            reward = 'Incorrect!'
    elif k[0][0]=='right':
        reward = 'Nothing happened!'
    else:
        reward = 'No response!'
        
    
    print(reward)
    keys.append(k)
    rewards.append(reward)
    
    rew.text = str(reward)
    rew.draw()
    win.callOnFlip(rewOnset.append,exptTime.getTime())
    win.flip()
    
    core.wait(0.1)

print(keys)
print(stim)
print(rewards)
print(fixOnset)
print(stimOnset)
print(promptOnset)
print(rewOnset)


