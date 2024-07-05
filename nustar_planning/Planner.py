import os
import jinja2
import pathlib
import astropy.units as u

from nustar_pysolar import planning, io 

from .Target import Target
from .Orbit import Orbit


def get_orbit_list(start: str, end: str) -> list[list]:

    # Get occultation period times
    fname = io.download_occultation_times(outdir='./data/')

    # Define observing window
    orbits = planning.sunlight_periods(fname, tstart=start, tend=end)

    return orbits


def coordinate_conversion(
    nu_intervals: list[list],
    nu_angles: list[u.Quantity],
    nu_centers: list[u.Quantity]
):
    """
    Perform a coordinate conversion using the provided inputs.
    """

    
    print("For a \"square\" field of view, use angle = 0 / 90 / 180 / 270 to have DET0 at the NE (top left) / SE / SW / NW\ncorners of a square field of view.")
    
    orbits = []
    for interval in nu_intervals:
        orbit_list = get_orbit_list(*interval)
        if orbit_list == -1:
            raise ValueError(f'No orbits found within time interval: {interval}')
        orbits += orbit_list

    # Loop over orbits and find pointing in RA/Dec
    # print(f'\nStarting @ {obs_starts[0]}')
    for i, orbit in enumerate(orbits):

        # Calculate the PA angle.
        pa = planning.get_nustar_roll(orbit[0], nu_angles[i].to(u.deg)) # angle is anti-clockwise starting with Det0 in top-left

        offset = nu_centers[i]
        midTime = (0.5*(orbit[1] - orbit[0]) + orbit[0])
        sky_pos = planning.get_skyfield_position(midTime, offset, load_path='./data', parallax_correction=True)
        print(f'\nOrbit: {i}')
        print(f'Orbit start: {orbit[0]} -> Orbit end: {orbit[1]}')
        print(f'Aim time: {midTime} RA: {sky_pos[0]}, Dec: {sky_pos[1]}')
        # print(f'NuSTAR Roll angle for anti-clockwise rotation of {nu_angles[i]} from SN @ {orbit[0]}: {pa}\n')
        print(f'Roll: {pa}\n')
    print('\n\n\n\n')


class Planner():


    def __init__(self, out_dir: str, orbits: list[Orbit] = []):
        
        self.out_dir = os.path.abspath(out_dir) + '/'
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        self._orbits = orbits
        if self._orbits:
            self._sort_orbits()


    @property
    def orbits(self) -> list[Orbit]:
        self._sort_orbits()
        return self._orbits


    @property
    def orbit_ids(self) -> list[str]:
        return [o.orbit_id for o in self.orbits]


    def _sort_orbits(self):

        # Sort orbits by time: https://stackoverflow.com/a/613218
        dt_times = {i: o.start.datetime for i, o in enumerate(self._orbits)}
        s = {k: v for k, v in sorted(dt_times.items(), key=lambda item: item[1])}
        self._orbits = [self._orbits[i] for i in s.keys()]

    
    def add_orbit(self, orbit: Orbit):
        
        if orbit.orbit_id not in self.orbit_ids:
            self._orbits.append(orbit)
        else:
            print(f'Orbit already in planner: {orbit.orbit_id}. Not adding')


    def add_orbits(self, orbits: list[Orbit]):

        for orbit in orbits:
            self.add_orbit(orbit)


    def plot_orbits(self):

        for i, orbit in enumerate(self.orbits, start=1):
            orbit.plot_targets(self.out_dir)

        
    def generate_report(self):

        latex_jinja_env = jinja2.Environment(
            block_start_string = '\BLOCK{',
            block_end_string = '}',
            variable_start_string = '\VAR{',
            variable_end_string = '}',
            comment_start_string = '\#{',
            comment_end_string = '}',
            line_statement_prefix = '%%',
            line_comment_prefix = '%#',
            trim_blocks = True,
            autoescape = False,
            loader = jinja2.FileSystemLoader(os.path.abspath('/'))
        )

        template_dir = os.path.dirname(__file__) + '/templates/'
        pdf_template = latex_jinja_env.get_template(f'{template_dir}pdf_template.tex')
        orbit_template = latex_jinja_env.get_template(f'{template_dir}orbit_template.tex')
        target_template = latex_jinja_env.get_template(f'{template_dir}target_template.tex')

        pdf_render = pdf_template.render(dict(
            cls_path=f'{template_dir}planning'
        ))
        renders = [pdf_render]

        for orbit in self.orbits:
            orbit_render = orbit_template.render(dict(
                orbit=orbit.orbit_id,
                start=orbit.time_range[0],
                end=orbit.time_range[1] 
            ))

            # renders = [orbit_render]
            renders.append(orbit_render)
            for target in orbit.targets:
                
                comments = ''
                if target.comments is not None:
                    comments = target.comments
                    if len(comments) > 4:
                        comments = comments[0:4]
                    comments = ' '*10 + ('').join(f'[{c}]' for c in comments)

                target_render = target_template.render(dict(
                    target=target.target_id,
                    start=target.time_range[0],
                    end=target.time_range[1],
                    coordinate=str(target.center),
                    image_file=target.file_path,
                    comments=comments
                ))

                    # target_render = target_render + comments

                renders.append(target_render)
            

        file_name = self.out_dir.split('/')[-2]
        report_file = f'{self.out_dir}/{file_name}.tex'

        with open(report_file, 'w') as tex_file:
            for render in renders:
                tex_file.write(render)
            tex_file.write('\n\end{document}')

        os.system('module load texlive')
        cmd = f'pdflatex -output-directory {self.out_dir} {report_file}'
        os.system(cmd)
        # subprocess.run(['module load texlive'], shell=True)
        # subprocess.run(['pdflatex', f'-output-directory {self.out_dir}', report_file], shell=True)

        
    def generate_timetable(self):

        orbits = self.orbits
        time_ranges = []
        for o in orbits:
            time_ranges.append(
                (
                    o.start - 15 * u.minute,
                    o.end + 15 * u.minute
                )
            )
        centers = [t.center for o in orbits for t in o.targets]
        angles = [t.angle for o in orbits for t in o.targets]

        coordinate_conversion(time_ranges, angles, centers)