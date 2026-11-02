"""Record an explicitly supplied operator answer; no model or simulation calls."""
from pathlib import Path
import argparse
from kyb.clarification import answer

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--answer',required=True)
    parser.add_argument('--scope',choices=['episode','same_public_contract'],default='episode',
        help='Broader reuse requires an explicit operator choice; default stays within this episode.')
    args=parser.parse_args()
    answer(args.run,args.answer,args.scope)
