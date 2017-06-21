from psychopy import visual, event, core

textHeight = 0.2

#create a visual window
win = visual.Window([400,400])

#create a text message
message1 = visual.TextStim(win, text = 'hello', height=textHeight)

#automatically draw every frame
#message.setAutoDraw(True)
message1.draw()

#flip the frame to the screen
win.flip()

#hold display for 2 seconds (better timing methods exist)
core.wait(2.0)

#make a new message and flip to the screen
message2 = visual.TextStim(win, text = 'world', height = 0.15)
message2.draw()

#rather than making a new message, change text and size of old message
#message1.text = 'world'
#message1.height = 0.15
#message1.draw()


win.flip()
core.wait(2.0)