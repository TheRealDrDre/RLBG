import os
import pickle
import random
from psychopy import visual, event, core
from psychopy.visual import ShapeStim
from psychopy.preferences import prefs
import pssThread


dataStream = pssThread.myThread(1,'dataStream',1)
dataStream.start()


fname = 'test'

fname = fname+'.pickle'

PSSdir = 'C:\\Users\\ausma_000\\Documents\\gp\\RLBG\\psypyPSS\\PSS'
#PSSdir = 'C:\\Users\\Experimenter\\Desktop\\psypyPSS\\PSS'

dataDir = 'C:\\Users\\ausma_000\\Desktop\\PSSdata\\singular'
#dataDir = 'C:\\Users\\Experimenter\\Desktop\\psypyPSS\\data\\singular'

#hiragana = 'C:\\Users\\ausma_000\\Desktop\\PSSdata\\fonts\\HIRAGANA.TTF'
hiragana = 'C:\\Users\\ausma_000\\Desktop\\PSSdata\\fonts\\CINEMATIME.TTF'
#hiragana = 'C:\\Users\\Experimenter\\Desktop\\psypyPSS\\PSS\\HIRAGANA.TTF'

os.chdir(PSSdir)

import PSSTask as pt

task = pt.PSS_Task()

trials = [element for tupl in task.TRAINING_BLOCK for element in tupl]
random.shuffle(trials)

total = 50

#create a visual window
#win = visual.Window([400,400],winType='pyglet',fullscr = True, monitor = None)
win = visual.Window([400,400],winType='pyglet',fullscr = False, monitor = 0)


vert = visual.Line(win,start = (0,-0.1),end = (0,0.1),lineWidth=2)
horz = visual.Line(win,start = (-0.1,0),end = (0.1,0),lineWidth=2)

stimulus = visual.TextStim(win, text = '', height = 0.3)
stimulus.fontFiles = [hiragana]
#stimulus.font = 'HIRAGANA'
stimulus.font = 'CINEMATIME'
prompt = visual.TextStim(win, text = '?')
prompt.height = 0.3
#prompt1 = visual.TextStim(win, text = 'Take it!', pos = (-.5, 0))
#prompt2 = visual.TextStim(win, text = 'Leave it!', pos = (.5, 0))
rew = visual.TextStim(win, text = '')
totalScore = visual.TextStim(win, text = '', pos = (0, -.5))
totalTitle = visual.TextStim(win, text = 'Total Score:', pos = (0, -.4))

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
    dataStream.event = 1
    core.wait(1.0)
    
    stim.append(t)
    stimulus.text = t
    stimulus.draw()
    win.callOnFlip(stimOnset.append,exptTime.getTime())
    win.flip()
    dataStream.event = 2
    core.wait(2.0)
    
    #prompt1.draw()
    #prompt2.draw()
    prompt.draw()
    win.callOnFlip(promptOnset.append,exptTime.getTime())
    win.flip()
    dataStream.event = 3
    
    count = 0
    k = ('','')
    while count < 1:
        k = event.waitKeys(maxWait = 2.0, keyList = ['left','right'], timeStamped=exptTime)
        dataStream.event = 4
        print(k)
        count =+ 1
    
    if k is None:
        k = [('',''),('','')]
    
    
    if k[0][0]=='left':
        value = task.get_reward(t)
        if value==1:
            reward = str(1)
            rew.color = 'green'
            total = total+1
            totalScore.color = 'green'
        elif value==-1:
            reward = str(-1)
            rew.color = 'red'
            total = total - 1
            totalScore.color = 'red'
    elif k[0][0]=='right':
        reward = str(0)
        rew.color = 'white'
        totalScore.color = 'white'
    else:
        reward = 'No response!'
        rew.color = 'white'
        totalScore.color = 'white'
        
    
    print(reward)
    keys.append(k)
    rewards.append(reward)
    
    rew.text = str(reward)
    rew.draw()
    totalScore.text = str(total)
    totalScore.draw()
    totalTitle.draw()
    win.callOnFlip(rewOnset.append,exptTime.getTime())
    win.flip()
    dataStream.event = 5
    
    core.wait(1.0)

print(keys)
print(stim)
print(rewards)
print(fixOnset)
print(stimOnset)
print(promptOnset)
print(rewOnset)

os.chdir(dataDir)

with open(fname, 'wb') as f:
    pickle.dump([keys, stim, rewards, fixOnset, stimOnset, promptOnset, rewOnset], f)


#to load the data:
#with open('test.pickle','rb') as f:
#    keys, stim, rewards, fixOnset, stimOnset, promptOnset, rewOnset = pickle.load(f)



