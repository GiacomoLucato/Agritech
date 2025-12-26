#!/usr/bin/env bash

notify-send -a ScopertaIDE -i $PWD/scoperta_ide/icon.png "Avvio di ScopertaIDE" "Verifica degli aggiornamenti in corso. Per favore, attendi..."

git fetch
git stash push -m "ScopertaIDE Updater Backup"
git reset --hard origin/main
git stash apply
git stash clear

scoperta_ide/scoperta_ide
