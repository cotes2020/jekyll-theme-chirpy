---
title: Pythagorean Tree
date: 2025-04-23 10:00:00 +0300
categories: [Dev, Design Patterns]
tags: [oop, singleton , factory, python, design patterns, numpy, opencv]
render_with_liquid: true
description: A visual and object-oriented implementation of the Pythagorean Tree using Python, showcasing design patterns like Singleton and Factory, and leveraging NumPy and OpenCV for graphical rendering.     

# image:
#   path: "/assets/img/images/PythagoreanTree/scene_night_moon.webp"
#   # lqip: data:image/webp;base64,UklGRuIAAABXRUJQVlA4INYAAACQBQCdASoUAAsAPm0skkWkIqGYBABABsS2AE6ZQjgbYBUoUWrZNBERDADEVLSVjE7Nl6ItAAD+/m7ELYcwg+RtFAPhHg3UNYByIGYlMyaubKcmMepfUAHyjEOKt4ydywBT871v5IS9b8FPdN/KfNOV/qx4wGk/0J6gv18RlA/b1qwgk90+GJhSyOas/3F077SHsj/evVy5Ul9F74NYlaCOMsCmshRz3NWhaxquuT7kq5W+r/dP+f/pk2Tx8v8yzLEtKds9ALqvFxls4/3PFwr/8dS3/AAA
#   alt: Responsive rendering of Chirpy theme on multiple devices.
light_image:
  path: /assets/img/images/PythagoreanTree/scene_day_sun.webp
  lqip: data:image/webp;base64,UklGRsoAAABXRUJQVlA4IL4AAADQBACdASoUAAsAPm0skkWkIqGYBABABsS2AE6ZQjubyX8VQDFW1YsmJWHTNWYHAAD++hR2pRF1Srf1BpL/priIV4qKWVMwv7qCayU/EI29KD3H1AfQOaGzFtisjJ/uS5hmORnfrhT+48wZ+UDurJBD/5UbyUYAUgqRYHvkd+VEe97suzqSjpmNObZ8LT2sd/8zOT1UTAQKn2QFL3tdoGaoj/H/P/280Pdkn9TuH7kuMqR7DuK01mjKjVZ/FwAA
  alt: Image for Light Mode
dark_image:
  path: /assets/img/images/PythagoreanTree/scene_night_moon.webp
  lqip: data:image/webp;base64,UklGRuIAAABXRUJQVlA4INYAAACQBQCdASoUAAsAPm0skkWkIqGYBABABsS2AE6ZQjgbYBUoUWrZNBERDADEVLSVjE7Nl6ItAAD+/m7ELYcwg+RtFAPhHg3UNYByIGYlMyaubKcmMepfUAHyjEOKt4ydywBT871v5IS9b8FPdN/KfNOV/qx4wGk/0J6gv18RlA/b1qwgk90+GJhSyOas/3F077SHsj/evVy5Ul9F74NYlaCOMsCmshRz3NWhaxquuT7kq5W+r/dP+f/pk2Tx8v8yzLEtKds9ALqvFxls4/3PFwr/8dS3/AAA
  alt: Image for Dark Mode
---

![Light mode only](/assets/img/images/PythagoreanTree/scene_day_sun.webp){: .light }
![Dark mode only](/assets/img/images/PythagoreanTree/scene_night_moon.webp){: .dark }

# Pythagorean Tree Generator

## Overview

This project generates visual scenes containing Pythagorean trees, using **OpenCV** and **NumPy**. Each scene can include a sky, earth, sun, moon, and stars, and multiple fractal trees with customizable colors and styles.

## Project Structure


```
PythagoreanTree/
├── main.py
├── tree/
│   ├── __init__.py
│   ├── generator.py
│   ├── drawer.py
│   ├── background.py
│   └── scene_generator.py
├──output/
│   ├── output.png
├── requirements.txt
├──README.md
└──.gitignore
```


## Libraries

- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

## Requirements

- Python 3.x
- Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage
To generate the Pythagorean tree, simply run:
```bash
python main.py
```
This will generate multiple images with different background 

### settings and tree designs.

Tree Customization
Each tree can be configured using the following parameters in `tree_gen.generate(...)`:
```python
generate(x, y, length, angle, branch_angle, depth)
```
### Tree Generation Parameters

- `x, y`: starting position of the tree  
- `length`: length of the base square  
- `angle`: starting angle (usually `π/2`)  
- `branch_angle`: angle between branches (e.g., `np.pi / 4`)  
- `depth`: recursion depth (e.g., `10`)  

### Tree Styling

Each tree dictionary also supports:

- `start_color`: RGB tuple for the base color  
- `end_color`: RGB tuple for the top color  
- `gradient`: one of `"linear"`, `"reverse"`, `"constant"`  

### Scene Configuration

Each scene is defined by a dictionary like:

```python
{
    "day": True or False,            # Background color (day/night)
    "sun": True or False,            # Show sun
    "moon": True or False,           # Show moon and stars
    "text": "Scene title",           # Text label
    "file": "output_file.png"        # Output file name
}
```

<!-- ### generator.py
<img src="output\Screenshot 2025-04-14 221645.png" alt="Alt Text" width="300"> -->

