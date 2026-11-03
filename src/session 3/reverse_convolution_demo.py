"""Try to work backwards: calculate the input image from the output of convolution_demo.py.

Run it from the top folder of the workshop repository with:

    uv run "src/session 3/reverse_convolution_demo.py"

Each output pixel is one equation ("weight × pixel + weight × pixel + ... = result").
To go backwards we solve those equations for the unknown input pixels.

The catch: convolution_demo.py gives only 9 output numbers, but the input has
25 pixels.  9 equations cannot pin down 25 unknowns, so the script tries two things:

    A. Only the 9 output numbers are known.
       -> numpy finds *a* solution, but it is not the original image.
    B. The 9 output numbers AND the 16 edge pixels of the input are known.
       -> now 9 equations for 9 unknowns: the original image comes back exactly.

The picture is saved as Pictures/convolution/reverse.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files only, no window on screen
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "Pictures" / "convolution"

# IMAGE and KERNEL must be the same as in convolution_demo.py.
# The IMAGE is only used to make the output numbers and to check our answer
# at the end -- the reconstruction itself never looks at it.
IMAGE = np.array(
    [
        [1, 2, 1, 0, 1],
        [2, 5, 6, 4, 1],
        [1, 6, 9, 5, 2],
        [0, 4, 5, 3, 1],
        [1, 1, 2, 1, 0],
    ]
)

KERNEL = [
    [None, 1, None],
    [1, 1, 1],
    [None, 1, None],
]

#KERNEL = [
#    [None, -1, None],
#    [-1, 4, -1],
#    [None, -1, None],
#]

N = 5
CENTERS = [(r, c) for r in range(1, N - 1) for c in range(1, N - 1)]  # the 9 inner pixels
EDGE = [(r, c) for r in range(N) for c in range(N) if (r, c) not in CENTERS]


def kernel_cells(row, col):
    """Image pixels covered by the structuring element centered at (row, col),
    with the weight of each one (same as in convolution_demo.py)."""
    cells = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            weight = KERNEL[dr + 1][dc + 1]
            if weight is not None:
                cells.append((row + dr, col + dc, weight))
    return cells


def convolve(image):
    """The forward step: the 9 output numbers, exactly as convolution_demo.py makes them."""
    return np.array([sum(w * image[r, c] for r, c, w in kernel_cells(row, col))
                     for row, col in CENTERS])


def equation_matrix():
    """Write all 9 equations as one table of weights (a 9 x 25 matrix).

    Row i holds the weights for output pixel i; column j belongs to input pixel j
    (pixels numbered row by row, 0 ... 24).  Then:  matrix @ input = output.
    """
    matrix = np.zeros((len(CENTERS), N * N))
    for i, (row, col) in enumerate(CENTERS):
        for r, c, w in kernel_cells(row, col):
            matrix[i, r * N + c] = w
    return matrix


def draw_grid(ax, values, title, highlight=None):
    """Draw a grid of numbers; pixels listed in `highlight` get a red background."""
    values = np.asarray(values)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            color = "#f4b6b6" if highlight is not None and highlight[r, c] else "#dbe9f6"
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#555"))
            text = f"{values[r, c]:.1f}".rstrip("0").rstrip(".")
            ax.text(c + 0.5, r + 0.5, text, ha="center", va="center", fontsize=13)
    ax.set_xlim(-0.1, values.shape[1] + 0.1)
    ax.set_ylim(values.shape[0] + 0.1, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)


def main():
    output = convolve(IMAGE)
    print("Output of convolution_demo.py (the green numbers):")
    print(output.reshape(3, 3), "\n")

    matrix = equation_matrix()
    print(f"We have {matrix.shape[0]} equations and {matrix.shape[1]} unknown input pixels.")
    unused = [(r, c) for r in range(N) for c in range(N) if not matrix[:, r * N + c].any()]
    if unused:
        print(f"These input pixels are never touched by the kernel, so nothing can be "
              f"learned about them: {unused}")

    # --- A. Only the output is known --------------------------------------------
    # lstsq finds the "smallest" image (closest to all zeros) that fits the equations.
    guess, *_ = np.linalg.lstsq(matrix, output, rcond=None)
    guess = guess.reshape(N, N)
    print("\nA. Best guess from the output alone:")
    print(np.round(guess, 2))
    print("Convolving the guess gives:", np.round(convolve(guess), 2),
          "-> the same output, but a different image!")

    # --- B. Output plus the edge pixels of the input are known --------------------
    # Move the known edge pixels to the right-hand side of each equation,
    # leaving 9 equations with 9 unknowns (the inner pixels).
    inner_cols = [r * N + c for r, c in CENTERS]
    edge_cols = [r * N + c for r, c in EDGE]
    edge_values = np.array([IMAGE[r, c] for r, c in EDGE])
    square = matrix[:, inner_cols]
    rhs = output - matrix[:, edge_cols] @ edge_values

    recovered = IMAGE.astype(float).copy()
    if np.linalg.matrix_rank(square) < len(CENTERS):
        print("\nB. With this kernel the 9 equations are not independent -- "
              "even knowing the edges, the inner pixels cannot be found uniquely.")
        recovered[1:-1, 1:-1] = np.nan
    else:
        inner = np.linalg.solve(square, rhs)
        recovered[1:-1, 1:-1] = inner.reshape(3, 3)
        print("\nB. Recovered image when the edge pixels are known as well:")
        print(np.round(recovered, 2))
        print("Same as the original input?", np.allclose(recovered, IMAGE))

    # --- Picture ----------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    draw_grid(axes[0], IMAGE, "Original input\n(hidden from the solver)")
    draw_grid(axes[1], output.reshape(3, 3), "Output of convolution_demo.py\n(what we know)")
    draw_grid(axes[2], guess, "A. Guess from output only\n(red = wrong)",
              highlight=~np.isclose(guess, IMAGE))
    draw_grid(axes[3], recovered, "B. Output + edge pixels known\n(red = wrong)",
              highlight=~np.isclose(recovered, IMAGE))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reverse.png", dpi=100)
    print(f"\nSaved picture to {OUT_DIR / 'reverse.png'}")


if __name__ == "__main__":
    main()
