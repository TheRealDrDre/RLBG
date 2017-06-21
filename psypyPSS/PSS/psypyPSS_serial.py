import os
from psychopy import visual, event, core
from psychopy.visual import ShapeStim

os.chdir('C:\\Users\\ausma_000\\Documents\\gp\\RLBG\\psypyPSS\\PSS')

import PSSTask as pt

task = pt.PSS_Task()

trials = task.TRAINING_BLOCK

print(task.state)

#create a visual window
win = visual.Window([400,400],winType='pyglet')

vert = visual.Line(win,start = (0,-0.1),end = (0,0.1),lineWidth=2)
horz = visual.Line(win,start = (-0.1,0),end = (0.1,0),lineWidth=2)

left = visual.TextStim(win, text = '', pos=(-0.5,0))
right = visual.TextStim(win, text = '', pos=(0.5,0))
#prompt = visual.TextStim(win, text = 'Left for first, right for second')
rew = visual.TextStim(win, text = '')

keys = list()
stim = list()
rewards = list()
fixOnset = list()
stim1Onset = list()
stim2Onset = list()
promptOnset = list()
rewOnset = list()

exptTime = core.Clock()

for t in trials:
    vert.lineColor = 'white'
    horz.lineColor = 'white'
    vert.draw()
    horz.draw()
    win.callOnFlip(fixOnset.append,exptTime.getTime())
    win.flip()
    core.wait(1.0)
    
    stim.append(task.state)
    left.text = task.state.left
    left.draw()
    vert.draw()
    horz.draw()
    win.callOnFlip(stim1Onset.append,exptTime.getTime())
    win.flip()
    core.wait(1.0)
    
    right.text = task.state.right
    right.draw()
    vert.draw()
    horz.draw()
    win.callOnFlip(stim2Onset.append,exptTime.getTime())
    win.flip()
    core.wait(1.0)
    
    #prompt.draw()
    vert.lineColor = 'red'
    horz.lineColor = 'red'
    vert.draw()
    horz.draw()
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
        value = task.get_reward(task.state.left)
        if value==1:
            reward = 'Correct!'
        elif value==-1:
            reward = 'Incorrect!'
    elif k[0][0]=='right':
        value = task.get_reward(task.state.right)
        if value==1:
            reward = 'Correct!'
        elif value==-1:
            reward = 'Incorrect!'
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
    
    task.state = task.next_state()
    
    print(task.state)
    
print(keys)
print(stim)
print(rewards)
print(fixOnset)
print(stim1Onset)
print(stim2Onset)
print(promptOnset)
print(rewOnset)




