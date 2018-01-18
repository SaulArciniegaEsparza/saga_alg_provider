"""
SAGA GIS algorithm provider
Environment management

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

import os
import subprocess
import time


class SAGAEnvironment:
    def __init__(self):
        """
        SAGA environment initialization
        """

        # Define attributes
        self.ovwlog = True     # overwritten log files
        self.saga_version = ''  # saga version
        self.stdlog = ''        # register log file
        self.errlog = ''        # error log file
        self.workdir = None     # work directory

        # Define work directory
        self.set_env(None)      # set environment
        self.set_workdir(None)  # set work directory

    def __repr__(self):
        return('SAGAEnvironment')

    def get_saga_version(self):
        return(self.saga_version)

    def set_env(self, env=None):
        """
        Set location of saga_cmd and get saga version
        INPUTS
         env      [string] environment path
        """

        # Define environment path
        if env is None:
            if os.name == "posix":  # Linux distribution
                env = "/usr/local/bin"

        if type(env) is str:
            # Check if path exist
            if not os.path.exists(env):
                raise IOError('Path "{}" does not exist!'.format(env))

            # Set environmental path
            os.environ["PATH"] += os.pathsep + env

            # Get SAGA version
            # create cmd
            cmd = ['saga_cmd', '--version']
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # run cmd
            text = p.communicate()[0]
            print(text)  # print saga version
            try:
                self.saga_version = text.split(':')[1].strip()[:5]
            except:
                self.saga_version = ''
                raise IOError('Environment path ({}) does not contains a saga_cmd'.format(env))

    def set_workdir(self, workdir=None):
        """
        Define work directory. Log and Error files will be stored in the workdir,
        also some temporary files are created in the workdir folder

        INPUTS
         workdir    [string] work directory
        """
        self.workdir = workdir
        if workdir is None:  # Set default workdir
            self.stdlog = "saga_processing.log"
            self.errlog = "saga_processing.error.log"

        elif type(workdir) is str:  # Input workdir
            # check if workdir exist
            if not os.path.exists(workdir):
                os.makedirs(workdir)  # create dir
            self.stdlog = os.path.join(workdir, "processing.log")
            self.errlog = os.path.join(workdir, "processing.error.log")

    def options(self, ovwlog=None):
        """
        Change SAGA algorithm provider environment options

        INPUTS:
           ovwlog       [bool] if True, saga outputs warning and messages are added to the
                          log and error files, In other case, these files are overwritten.
                          By default, no changes are made.
        """
        if type(ovwlog) is bool:
            self.ovwlog = ovwlog

    def print_commands(self, library=None, tool=None):
        """
        Prints a list of SAGA GIS commands given a library name and tool id
        If no inputs are used, all libraries are printed.
        If only library is input, all tools in these library is printed

        INPUTS
         library  [string]  library name. If library is None, then a list of
                            libraries is printed
         tool     [int] tool number in library name. If tool is None, then all tools
                        in library are printed
        """
        # Check input library
        if library is None:
            cmd = ['saga_cmd']
        elif library is not None and tool is None:
            cmd = ['saga_cmd', '-f=q', str(library)]
        else:
            cmd = ['saga_cmd', '-f=q', str(library), str(tool)]
        # Open commands virtual file
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # run cmd
        text = p.communicate()[0]
        print(text)  # print algorithms

    def runalgorithm(self, library, tool, parameters):
        """
        Run command tool using library name and tool number

        INPUTS
         library      [string] library name
         tool         [int] tool number
         parameters   [dict] input parameters
        """
        # Check inputs
        if type(library) is not str:
            raise TypeError('Wrong library parameter!')
        if type(tool) is not int:
            raise TypeError('Wrong tool parameter!')
        if type(parameters) is not dict:
            raise TypeError('Wrong parameters variable type!')
        # Create executable cmd
        cmd = ['saga_cmd', '-f=q', library, str(tool)]
        for key, param in parameters.iteritems():
            cmd.extend(['-' + str(key).upper(), str(param)])
        # Run command
        self.run_command_logged(cmd);

    def run_command_logged(self, cmd):
        """
        Run command using the saga_cmd

        OUTPUTS
         flag    [boolean] if cmd is executed then flag is True
        """
        # Initial condition
        flag = True
        # Open log files
        if self.ovwlog:
            logstd = open(self.stdlog, "a")
            logerr = open(self.errlog, "a")
        else:
            logstd = open(self.stdlog, "w")
            logerr = open(self.errlog, "w")
        # Initial time
        t0 = time.time()
        # Run command
        try:  # try to run
            subprocess.call(cmd, stdout=logstd, stderr=logerr)
        except Exception, e:
            logstd = open(self.stdlog, "a")
            logerr = open(self.errlog, "a")
            logerr.write("Exception running: {} {}".format(cmd[2], cmd[3]))
            logerr.write("ERROR {}\n".format(e))
            logstd.close()
            logerr.close()
            print("Error running: {} {}".format(cmd[2], cmd[3]))
            flag = False
            return(flag)
        # Close files
        logstd.write("\n\nProcessing finished in " + str(int(time.time() - t0)) + " seconds.\n")
        logstd.close()
        logerr.close()
        return(flag)
        # End run_command_logged()

