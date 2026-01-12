#!/bin/bash
cd f:\\github\\financial_agent_fork\\desktop_app
rm -r node_modules -Force -ErrorAction SilentlyContinue
npm install
npm run dist-win
