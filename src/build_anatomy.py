#!/usr/bin/env python3
import os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/viz'
tpl = open(f"{H}/anatomy_template.html").read()
out = tpl.replace("/*__DATA__*/", open(f"{H}/neurons.json").read())
open(f"{H}/anatomy.html","w").write(out)
print(f"  viz/anatomy.html  {len(out)/1024:.0f} KB")
