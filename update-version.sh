#!/bin/bash

#Ensure that the local repository is up to date with the one on github.com

git pull origin master

# Save the current date and time in a file called version

date > version

# Add the file to the repository

git add version

#Commit the changes

git commit -m "Update version"

# Push the changes to the remote repository

git push origin master
