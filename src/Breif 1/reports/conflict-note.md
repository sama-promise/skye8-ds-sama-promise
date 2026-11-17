# Merge Conflict Note

For Stage A I had to create a merge conflict on purpose and then fix it, so here's what I did and what happened.

I made two branches, branch-x and branch-z. On branch-x I added a line to README.md saying who maintains the project. I merged that into main and it went in fine, no problems.

Then I made branch-z, but I made it start from an earlier point (before branch-x existed), and on this branch I changed the exact same line in README.md but wrote something different.

When I tried to merge branch-z into main, git couldn't figure out which version of that line to keep since both branches had changed it differently. It gave me this error:

CONFLICT (content): Merge conflict in README.md

When I opened the file, git had put in these markers around the two different versions:

<<<<<<< HEAD
(the version already on main)
=======
(the version from branch-z)
>>>>>>> branch-z

To fix it I just read both versions and combined them into one sentence that made sense, then deleted the <<<<<<<, =======, and >>>>>>> lines completely so only my final version was left. After that I saved the file, ran git add, git commit, and git push and the merge went through fine.

Final line I ended up with:
"Maintained by Sama Promise Njemchama as part of the Skye8 Data Science internship programme."