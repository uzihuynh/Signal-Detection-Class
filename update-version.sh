#!/bin/bash

#Ensure that the local repository is up to date with the one on github.com

git pull origin main

# Save the current date and time in a file called version

date > /data/class/cogs106/tommieh/cogs106-tommie/version

# Add the file to the repository

git add .

#Commit the changes

git commit -m "Updated version"

# Push the changes to the remote repository

git push --set-upstream  origin main
