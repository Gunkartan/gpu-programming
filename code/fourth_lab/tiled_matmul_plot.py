import matplotlib.pyplot as plt

if __name__ == '__main__':
    configurations = [
        'Tile, 8, N, 2048',
        'Tile, 16, N, 2048',
        'Tile, 32, N, 2048',
        'Tile, 16, N, 4096'
    ]
    gflops = [
        2470.80,
        3514.73,
        3308.33,
        3772.38
    ]
    naive_gflops = 2564.46
    plt.figure(figsize=(
        10,
        6
    ))
    bars = plt.bar(
        configurations,
        gflops
    )
    plt.axhline(
        naive_gflops,
        color='red',
        linestyle='--',
        linewidth=2
    )
    plt.text(
        3.45,
        naive_gflops + 50,
        f'The naive baseline is {naive_gflops:.2f} GFLOPS',
        ha='right',
        va='bottom',
        color='red',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            pad=3
        )
    )
    plt.xlabel('Configurations')
    plt.ylabel('GFLOPS')
    plt.title('Tiled matrix multiplication performance')

    for bar, value in zip(
        bars,
        gflops
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.2f}',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig(
        'tiled_plot.png',
        dpi=300
    )
    plt.show()