---
title: "git command line documentation"
date: 2018-08-29T09:55:50+02:00
categories:
- git
- github
tags:
- git
- github
keywords:
- git
- data science
- github
comments:       true
showMeta:       true
showActions:    true
thumbnailImage: https://png.icons8.com/windows/1600/git.png
---

<!-- toc -->


# IDE vs Shell
So far I was not happy with all the IDEs (Rstudio/Gitkraken/Github desktop) that I have tested. They provide some convenient shortcuts but in the end they tended to screw things up for me. That is why I am writing this documentation.

## Advantages of IDEs
IDEs are much better at visualizing the branches and the commits. They are also much better for resolving merge conflicts (gitkraken).

## Disadvantages of IDEs
You are always missing a functionality that makes you switch to the Shell. It is better to be literate with the shell commands all together to speed up your workflow.

# The git repository
The git repository is your project folder, it contains some hidden files and folders that are used by git to track your commits and the branches. If you run git and open your repository you can set the folder to mirror commits or branches. This will actually change the files that you can find with your system explorer.



# Introduce yourself

```{}
git config --global user.name 'Jennifer Bryan'
git config --global user.email 'jenny@stat.ubc.ca'
git config --global --list
```

# Configure Proxy

```{}
git config --global http.proxy http://@173.15.80.153:8080
```
# Setup local git without remote repos and RStudio

start with

```{}
git init
```

setup .gitignore (see instruction further down)

```{}
git add .
git commit -m 'message'
```


# Setup local git repository with remote repository

## starting with a local repository

- Setup a new project using the git option in Rstudio
- Create a new git repository on the github webpage

### add the remote repository

```{}
git remote add origin https://github.com/user/repo.git
```
### verify
```{}
git remote -v
```
### Connect the two master branches from local and remote
```{}
git push -u updateR master -f
```

## starting with a remote repository

open the directory you want to clone the repository in right click and select `git bash here`

```{}
git clone https://github.com/erblast/oetteR.git
```
# .gitignore

The `.gitignore` file contains a list of filenames that will be ignored and not and not tracked by git. When you are starting a git project it easy to skip this important file. It is however crucial to set it up before you start coding away. Once you are already deep in the project your workflow will be seriously disturbed when you keep having to do commits for file changes that you cannot control (for example `thumbs.db`) or have to resolve merge conflicts in outputfiles of your code such as `*.pdf` and `*.html` files. You will find that adding filenames to `.gitignore` that are already tracked will not work. You have to tediously add them manually in your git bash shell.

## setting up `.gitignore`

- Creating your git project with Rstudio will already add some R/Rstudio files to `.gitignore`.  
- [Here](https://github.com/github/gitignore/blob/master/R.gitignore) you can find a useful list of files to be added for R projects.  

## Creating `.gitignore` from scratch
windows will not let you create files with a file name starting with '.'
use the windows command line tool

```{}
rename gitignore.txt .gitignore
```

## Untracking files in hindsight

**untracking a single file**    
```{}
git rm --cached [file_name]
```

**untracking all files in a directory**
```{}
git rm --cached inst/\*
```

**untrack all files of a certain type**  
```{}
git rm -r --cached **/*.html
```

# Navigating in the shell

**switch between branches**
remember you create them only in the remote/origin repository
```{}
git checkout branch
```

**what's going on?**
Carefull the status is not always up to date. It seems to not check if the branch was updated from somewhere else than the current computer. It hapenned several times that the reply to the command implied that the current branch is up-to-date with the origin branch but then a pull command would still retrieve changes.
```{}
git status
```

## Pull

**Pull all**  
this would be nice to have but it DOES NOT WORK, it fetches from all branches but MERGES ONLY THE ACTIVE BRANCH.
```{}
git pull --all
```
**pull active branch**
```{}
git pull
```
**pull and rebase**  
If you expect changes in you remote repository for your active branch (if your working on a collaborative project), you can rebase before/while pulling.
```{}
git pull --rebase
```
see http://kernowsoul.com/blog/2012/06/20/4-ways-to-avoid-merge-commits-in-git/, for setting this up permanently

## commit
Use `git status` to see a if there are any changes to commit. Commit as often as you like, give resonable namings to you commits. A commit will not work if you do not supply a commit message. Git will let you rollback to any commit you made. Always use the `-m "text"` operator when calling `git commit`, otherwise you will end up will display confusing commit screen.

```{}
git commit -m "enter message here"
```

### if you see the confusing commit screen
you forgot to type `-m "text"`

**hit `ESC` **
```{}
:wq
```
### git started tracking changes that are meaningless and annoy me
reset all changes
```{}
git reset HEAD --hard
```
### i screwed up and made some stupid commits that i want to remove
**Opt1: roll back to remote branch version**
```{}
git reset --hard origin/workbranch
```

**Opt2: undo last commit**  
```
git reset HEAD~
```
### Should I amend to previous commits?
No, dont do it. This only work if you have not pushed after your previous commit. If you have pushed you possibly get a merge conflict between origin and the local repository because the two commits are different. You can however fix this by rebasing the remote branch.

```{}
git rebase origin/master
```

## Push
Push local commits to remote

```{}
git push
```

Overwrite remote version
```{}
git push origin branch_name -f
```

## Rebase
rebase to update your work branch from master DO NOT FORGET MASTER in the command. Your branch will not fully merge and stuff will be missing and you will get merge conflicts.
**rebase active branch**
```{}
git rebase master
```
## Merge
Merging can be the most annoying part of using git, I tend to have merge conflicts all the time and they are annoying to resolve. I will have an extra section on this further down.

**merge branch into master**
```{}
git checkout master
git merge work_branch
```
## Merge conflicts
Those are tedious to resolve and eat up time that could better be spent coding. If you have them use `git status` to see whats wrong. Open the files with an editor and find the highlighted conflicting code and edit it. (remove the highlights inserted by git and keep the code that is correct). If there is an delete you can either add `git add filename` or remove (`git rm filename`). I think it is best to use this manual approach for a while and develope a disciplined git workflow that makes you avoid these all together. Then when you think you got it right switch to gitkraken to handle your merge conflicts.

## Restore old commit versions
Checkout commit history of a certain branch
```{}
git log master

git revert 2342jk23l4jl2hg23k4j2h34
```
# Managing branches

## Create branch
```{}

git checkout -b [name_of_your_new_branch]

```

## Delete branch

```{}

git branch -d [name_of_your_new_branch]

```

## Accidently worked on the wrong branch
Sometimes you want to commit your changes and realize that you were working in master instead of your release or feature branch
```
git stash
git checkout branch123
git stash apply
```

# Versioning

Your `master` branch should always be the branch carrying the latest stable version. Branching off `master` you have a `release` branch  which carries the newest developments. From this you branch off the other branches working on different aspects of the next release. Once the `release` branch is stable it can be merged with `master`

# Remove files from the history

Sometimes you might need to remove files from prior commits, either you carelessly commited large binary files which are bloating up your history, or some of the history contains sensitive information like passwords.

This is an incredible hard thing to do. You will find lots of online ressources the most common being a command called `filter-branch`. However removing files from git like this is a bit like rying to uninstall a program by removing the `run.exe` from the installation folder. There will be a  lot of clutter and references which will not be removed. I recommend the following in this case.

- remove the files you want to remove from your current branch
- create an on-disk copy of your repository
- delete your branch locally and from the remote
- create a new branch
- restore your branch by copying all files from your on-disk copy
- commit all changes
- push to remote

You cannot remove files from the master branch like this. Either if you do not need the history start a new repo, or figure out the `filter-branch` command. This however needs some deeper knowledge on how `git` acutally works.

# Submodules
SUbmodules are great if you need to include code from another repository into your current repository that should be kept up to date.

```
git submodule add repo_adress
git submodule init
git submodule update
```

## Remove Submodules
If you dump another git repository into your currrent git repository because you might want to merge the two. Git will create a submodule that persits even if you delete all the git files from the repo that you want to fuze.

```
git rm --cached submodule_path # delete reference to submodule HEAD (no trailing slash)
git rm .gitmodules             # if you have more than one submodules,
                               # you need to edit this file instead of deleting!
rm -rf submodule_path/.git     # make sure you have backup!!
git add submodule_path         # will add files instead of commit reference
git commit -m "remove submodule"
```

# Release Tags
It is good practice to add a release tag after bumping up the version of your repository. It is the easiest to to this via the github website interface. However if you made a mistake and want to remove your current release because you made a typo github will not remove your release tag entirely but simply mark it as deleted which prevents you from creating a new tag with the same name. Thus we can use the following console command to delete it completely ([source](https://gist.github.com/mobilemind/7883996)).

```
# delete local tag '12345'
git tag -d 12345
# delete remote tag '12345' (eg, GitHub version too)
git push origin :refs/tags/12345
```
