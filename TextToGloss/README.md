# PharmaSign Text → Gloss

## What it does
Reads an Excel file (`Instructions.xlsx`) that contains a column named **كلام الصيدلي** and generates:
- `pharmasign_text2gloss.csv`
- `pharmasign_text2gloss.json`
- `pharmasign_text2gloss.xlsx`

## How to run
```bash
pip install -r requirements.txt
jupyter notebook pharmasign_text2gloss_clean.ipynb
```

## Input
Place `Instructions.xlsx` in the same folder as the notebook (or change `XLSX_PATH`).

## Notes
- Slot extraction rules are heuristic (V1) and easy to extend.
