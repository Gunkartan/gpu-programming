import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.DataFrame({
        'block_size': [
            32,
            64,
            128,
            256,
            512,
            1024
        ],
        'occupancy': [
            50.00,
            100.00,
            100.00,
            100.00,
            100.00,
            66.67
        ],
        'bandwidth': [
            360.88,
            357.73,
            355.98,
            356.56,
            391.26,
            394.68
        ]
    })
    fig, ax1 = plt.subplots()
    ax1.plot(
        data['block_size'],
        data['bandwidth'],
        marker='o',
        color='blue',
        label='Measured bandwidth'
    )
    ax1.set_xlabel('Block size in threads per block')
    ax1.set_ylabel('Measured bandwidth in GB per second')
    ax1.set_xscale(
        'log',
        base=2
    )
    ax1.set_xticks(data['block_size'])
    ax1.set_xticklabels(data['block_size'])
    ax2 = ax1.twinx()
    ax2.plot(
        data['block_size'],
        data['occupancy'],
        marker='s',
        color='orange',
        label='Theoretical occupancy'
    )
    ax2.set_ylabel('Theoretical occupancy in %')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2
    )
    plt.title('Block size vs bandwidth and occupancy for SAXPY')
    plt.tight_layout()
    plt.show()