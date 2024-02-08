import numpy as np
import matplotlib.pyplot as plt
from pipe_flow_eqs import get_local_Mach

cmap = plt.cm.jet
# Create a ScalarMappable and initialize a data structure for the color data
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=2))
# Add the colorbar to the figure

def apply_colors_mach(U,patches,jump):

    mach = get_local_Mach(U)
    mach_to_plot = mach[::jump]
    colors = cmap(mach_to_plot)
    colors = sm.to_rgba(mach_to_plot)
    # Extract RGB values from the colors array
    rgb_values = colors[:, :3]  # assuming cmap returns RGBA, we take only RGB

    # Update edgecolor and facecolor of all patches at once
    for patch, color in zip(patches, rgb_values):
        patch.set_edgecolor(color)
        patch.set_facecolor(color)
        patch.set_linewidth(0)  # Set linewidth to 0 if you want to remove the border

    # sm.set_clim(vmin=np.min(colors), vmax=np.max(colors))
    # cbar.update_normal(sm)


def setup_plot(x,S,dx,U,N, N_toplot = 200):
    # # Setup Plotting
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].set_title("Mach Number")
    # line_mach, = axs[0,0].plot(x,get_local_Mach(U), color = 'red')
    axs[0, 0].plot(x, S, label="S(x)", color="black")
    axs[0, 0].plot(x, -S, color="black")
    cbar = fig.colorbar(sm, ax=axs[0, 0])

    jump = int(N/N_toplot)
    # Create N patches once and add them to the plot
    patches = [
        plt.Rectangle((x[jump*i], -S[jump*i]), jump * dx[jump*i], 2 * S[jump*i], color="none") for i in range(int(N_toplot))
    ]
    for patch in patches:
        axs[0, 0].add_patch(patch)

    apply_colors_mach(U,patches,jump)

    axs[1, 0].set_title("Rho")
    (line_rho,) = axs[1, 0].plot(x[::jump], U[0, ::jump], color="orange")

    axs[0, 1].set_title("U")
    (line_u,) = axs[0, 1].plot(x[::jump], U[1, ::jump], color="blue")

    axs[1, 1].set_title("P")
    (line_p,) = axs[1, 1].plot(x[::jump], U[2, ::jump], color="black")
    fig.tight_layout()

    fig.suptitle(f"Time: {0:.4f}")
    fig.show()
    return fig, axs, line_rho, line_u, line_p, patches

def update_plot(U,t, fig, axs, line_rho, line_u, line_p, patches,N_toplot = 200):
    jump = int(U.shape[1]/N_toplot)
    fig.canvas.flush_events()
    line_rho.set_ydata(U[0, ::jump])
    line_u.set_ydata(U[1, ::jump])
    line_p.set_ydata(U[2, ::jump])
    fig.suptitle(f"Time: {t:.4f}")
    apply_colors_mach(U,patches,jump)
    fig.canvas.draw_idle()

    for ax in axs.flatten():
        ax.relim()
        ax.autoscale_view()


def setup_plot2(x,S,dx,U,N, N_toplot = 200):
    # # Setup Plotting
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    jump = int(N/N_toplot)
    axs[0, 0].set_title("Mach Number")
    line_mach, = axs[0,0].plot(x[::jump],get_local_Mach(U)[::jump], color = 'red')

    axs[1, 0].set_title("Rho")
    (line_rho,) = axs[1, 0].plot(x[::jump], U[0, ::jump], color="orange")

    axs[0, 1].set_title("U")
    (line_u,) = axs[0, 1].plot(x[::jump], U[1, ::jump], color="blue")

    axs[1, 1].set_title("P")
    (line_p,) = axs[1, 1].plot(x[::jump], U[2, ::jump], color="black")
    fig.tight_layout()

    fig.suptitle(f"Time: {0:.4f}")
    fig.show()
    return fig, axs,line_mach, line_rho, line_u, line_p

def update_plot2(U,t, fig, axs, line_mach, line_rho, line_u, line_p,N_toplot = 200):
    jump = int(U.shape[1]/N_toplot)
    fig.canvas.flush_events()
    line_mach.set_ydata(get_local_Mach(U)[::jump])
    line_rho.set_ydata(U[0, ::jump])
    line_u.set_ydata(U[1, ::jump])
    line_p.set_ydata(U[2, ::jump])
    fig.suptitle(f"Time: {t:.4f}")
    
    fig.canvas.draw_idle()

    for ax in axs.flatten():
        ax.relim()
        ax.autoscale_view()