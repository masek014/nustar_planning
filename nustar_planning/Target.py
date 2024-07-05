import datetime
import astropy.time
import astropy.units as u

from dataclasses import dataclass

from . import mapping as maps


@dataclass
class Target():
    target_id: str
    time_range: tuple[astropy.time.Time]
    center: u.Quantity
    fov: u.Quantity
    angle: u.Quantity
    psp: list[tuple] = None
    pois: list[tuple] = None
    aia_wavelengths: tuple[u.Quantity] = (94, 131) * u.angstrom
    # stereo_wavelength: u.Quantity = 195*u.angstrom
    comments: list[str] = None


    @property
    def start(self) -> astropy.time.Time:
        return self.time_range[0]
    

    @property
    def end(self) -> astropy.time.Time:
        return self.time_range[1]
    

    # TODO: Phase out the use of datetime
    @property
    def midtime(self) -> datetime.datetime:

        duration = (self.end - self.start).to(u.second)
        assert duration > 0, f'Target \'{self.target_id}\' observation duration should be a positive value.'
        mid_dt = (self.start + duration/2).datetime

        return mid_dt.strftime(maps.DT_TIME_FORMAT)

    
    def plot(self, out_dir: str = None) -> tuple:
        """
        Creates plot of the AIA and STEREO plot of a specific time projected onto a future time.
        
        Returns
        -------
        Tuple of axes for each subplot created.
        """

        fig = maps.plt.figure(figsize=(12,12), layout='constrained')
        fig.suptitle(f'{self.orbit.orbit_id} - {self.target_id} Projection')
        gs = fig.add_gridspec(2, 2,
            left=0.1, right=0.9,
            bottom=0.1, top=0.9,
            wspace=0.15, hspace=0.075
        )
        axs = []

        # satellites = ['aia', 'stereo']
        # wavelengths = [self.aia_wavelength, self.stereo_wavelength]
        for i, wavelength in enumerate(self.aia_wavelengths):

            sat = 'aia'
            satmap = maps.most_recent_map(sat, wavelength)
            projected_satmap = maps.project_map(satmap, self.midtime)

            if sat == 'aia':
                cmap = (satmap.cmap).reversed()
            else:
                cmap = 'YlGn'

            ax1 = fig.add_subplot(gs[i,0], projection=satmap)
            maps.plot_map(satmap, ax1, cmap,
                title=f'Original {sat.upper()} Map\n{sat.upper()} ' +
                    f'{wavelength.value:.0f} {satmap.date}')

            ax2 = fig.add_subplot(gs[i,1], projection=projected_satmap)
            maps.plot_map(projected_satmap, ax2, cmap,
                title=f'Reprojected to an Earth Observer\n{sat.upper()} ' +
                    f'{wavelength.value:.0f} {projected_satmap.date}')

            maps.draw_nustar_fov(projected_satmap, ax2, self.center, self.angle,
                self.fov, pixscale=satmap.scale[0])

            if self.psp is not None:
                for coord in self.psp:
                    maps.mark_psp(coord, ax=ax2, frame=projected_satmap)
            
            if self.pois is not None:
                for coord in self.pois:
                    maps.mark_poi(coord, ax=ax2, frame=projected_satmap)

            axs += [ax1, ax2]

        if out_dir is not None:
            if hasattr(self, 'orbit'):
                orbit_id = (self.orbit.orbit_id).lower().replace(' ', '') + '_'
            else:
                orbit_id = ''
            target_id = (self.target_id).lower().replace(' ', '')
            date = self.time_range[0].strftime('%Y-%m-%d')
            self.file_path = f'{out_dir}/{date}_{orbit_id}{target_id}.png'
            maps.plt.savefig(self.file_path, dpi=200)

        return fig, axs
