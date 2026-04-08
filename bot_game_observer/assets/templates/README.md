# Template images

Place small PNG crops here (grayscale matching works best):

- `spin_button_ready.png` — spin control in its clickable state  
- `popup_close_x.png` — close control for blocking popups  
- `win_banner.png` — win presentation area  
- `bonus_tease.png` / `bonus_trigger.png` — game-specific bonus cues  
- `near_miss.png` — optional cue for near-miss (highly game-specific)  
- `scatter_symbol.png` — symbol crop used for settled-frame scatter counting
- `bonus_symbol.png` — symbol crop used for settled-frame bonus counting
- `session_end.png` — optional “session over” indicator  

Use `python calibrate.py` to capture crops from the live window. Tune `threshold` values in `config/default.yaml` per asset.
