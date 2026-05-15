

# `app/theme/css.py`

## Purpose
This file defines the app’s visual theme system and generates the CSS injected into the Streamlit UI. It centralizes theme colors, fonts, spacing, and shell/component styling so the app has a consistent, polished look.

## Required imports
None.

The current `css.py` implementation does not require any Python imports.

## Main contents

### `THEMES`
A dictionary of theme presets, currently:
- `dark`
- `light`

Each theme contains reusable design tokens such as:
- background, card, and surface colors
- border and glow colors
- accent colors
- text and muted colors
- gradients
- shadows
- chart palette values
- chat/status UI colors

### `_rgb(hex_color: str) -> str`
Small helper that converts a hex color like `#38bdf8` into an `r,g,b` string for use with `rgba()`.

Example:
- input: `#38bdf8`
- output: `56,189,248`

### `build_css(t: dict, dark: bool) -> str`
Builds and returns the full CSS string for the selected theme.

Parameters:
- `t`: a theme dictionary from `THEMES`
- `dark`: whether the selected theme is dark mode

The generated CSS styles the main app shell, including:
- page background and typography
- Streamlit container
- sidebar
- hero section
- cards/panels
- tabs
- inputs and buttons
- chat panel
- badges/status pills
- expanders
- scrollbar

## Typical usage
1. Select a theme from `THEMES`
2. Call `build_css(...)`
3. Inject the returned CSS into Streamlit from a loader/helper file

## `requirements.txt`
No extra package is needed specifically for `css.py`.

Relevant requirement:
```txt
streamlit
```

Nothing else is required for this file.