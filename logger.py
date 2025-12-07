#####################################
# Class for handling logging
#####################################

import sys
import time
from socket import gethostname

class Logger:
  def __init__(self, logfilename):
    self.logfilename = logfilename
    self.logfile = open(logfilename,"w")
    self.screen = 1
    self.log = 1
    localtime= time.asctime(time.localtime(time.time()))
    self.joint("starting log: %s\n" % localtime)
    self.joint("running on: " + gethostname() + "\n\n")

  def closelog(self):
    localtime= time.asctime(time.localtime(time.time()))
    self.joint("\nclosing log: %s\n" % localtime)
    self.logfile.close()

  def joint(self, mystring, *args):
    write = 1
    for arg in args:
        write = arg
    if self.log and write==1:
      self.logfile.write(mystring)
      self.logfile.flush()
    if self.screen and write==1:
      print(mystring, end=''),
      sys.stdout.flush()
      
  def both_on(self):
    self.screen = self.log = 1
    
  def both_off(self):
    self.screen = self.log = 0
    
  def screen_on(self):
    self.screen = 1

  def screen_off(self):
    self.screen = 0

  def log_on(self):
    self.log = 1

  def log_off(self):
    self.log = 0

  def stateandquit(self, stuff):
    self.log = self.screen = 1
    self.joint(stuff + "\n")
    self.closelog()
    sys.exit("\nquitting\n")

  def get_filename(self):
    return self.logfilename
