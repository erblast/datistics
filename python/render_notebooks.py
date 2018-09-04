

from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor
from pathlib import Path
from traitlets.config import Config

import nbformat
import re

path = r'./notebooks/adding_posts_with_juypter.ipynb'

# here I customize some functions of the fabfile.py of the hugo_jupyter package

class CustomPreprocessor(Preprocessor):
    """Remove blank code cells and unnecessary whitespace."""

    def preprocess(self, nb, resources):
        """
        Remove blank cells
        """
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and not cell.source:
                nb.cells.pop(index)
            else:
                nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
        return nb, resources

    def preprocess_cell(self, cell, resources, cell_index):
        """
        Remove extraneous whitespace from code cells' source code
        """
        if cell.cell_type == 'code':
            cell.source = cell.source.strip()

        return cell, resources


def doctor(string: str) -> str:
    """Get rid of all the wacky newlines nbconvert adds to markdown output and return result."""
    post_code_newlines_patt = re.compile(r'(```)(\n+)')
    inter_output_newlines_patt = re.compile(r'(\s{4}\S+)(\n+)(\s{4})')

    post_code_filtered = re.sub(post_code_newlines_patt, r'\1\n\n', string)
    inter_output_filtered = re.sub(inter_output_newlines_patt, r'\1\n\3', post_code_filtered)

    return inter_output_filtered

def remove_java_script(x):
    """strip java script from md file"""

    script_regex1 = re.compile("<script type='text/javascript'>[\s\S]*?\{window\.Plotly = Plotly;\}\);\}</script>")
    script_regex2 = re.compile("<div id=[\s\S]*?plot.ly\"}\)}\);</script>")

    x = re.sub(script_regex1, '', x)
    x = re.sub(script_regex2, "", x)

    return x

def make_yaml_header(**kwargs):

    header = '---\n'

    for key, value in kwargs.items():
        if type(value) in [str, int , float ] :
            header += '{} : {}\n'.format(key, value)
        else:
            header += '{}: \n'.format(key)
            for item in value:
                header += '  - {}\n'.format(item)

    header += '---\n'

    return header


def notebook_to_markdown( path, date, slug, **kwargs ):
    """
    Convert notebook to Markdown format

    Args:
        path: str, path to notebook
        date: datestring in YYYY-MM-DD format
        slug: str, front-matter parameter, used to compose adress of blogpost
        kwargs: str, float, int, list, tuple, other front-matter parameters recommended to pass title

    """

    path_nb = Path(path)
    path_out = path_nb.parents[1] / 'static'/ date.split('-')[0] / date.split('-')[1] / slug
    path_post = path_nb.parents[1] / 'content/post/' / ( date + '-' + slug + '.md' )



    assert path_nb.exists()
    assert path_post.parent.exists()
    assert bool( re.match('[0-9]{4}-[0-1][0-9]-[0-3][0-9]', date) ), 'Incorrect date format, need YYYY-MM-DD'

    # convert notebook to .md----------------------------------------------------

    with Path(path).open() as fp:
        notebook = nbformat.read(fp, as_version=4)

    c = Config()
    c.MarkdownExporter.preprocessors = [CustomPreprocessor]
    markdown_exporter = MarkdownExporter(config=c)

    markdown, resources = markdown_exporter.from_notebook_node(notebook)
    md = doctor(markdown)
    md = remove_java_script(md)

    yaml = make_yaml_header(  date = date
                             , slug = slug
                             , **kwargs)

    md = yaml + md

    with path_post.open('w') as f:
        f.write(md)


    # write outputs as png --------------------------------------------------------

    if 'outputs' in resources.keys():
        if not path_out.exists():
            path_out.mkdir(parents=True)

        for key in resources['outputs'].keys():
            with (path_out / key).open('wb') as f:
                f.write( resources['outputs'][key] )



if __name__ == "__main__":

    path = r'../notebooks/adding_posts_with_juypter.ipynb'

    notebook_to_markdown( path = r'../notebooks/adding_posts_with_juypter.ipynb'
                          , date = '2018-08-11'
                          , slug = 'post_with_jupyter'
                          , title = 'Use jupyter notebooks to add posts to hugo blog'
                          , author = 'Bjoern Koneswarakantha'
                          , categories = ['R vs. python','blogging']
                          , tags = ['R vs. python','hugo', 'python', 'R', 'hugo', 'hugo_jupyer', 'jupyter']
                          , summary = 'Blogging with jupyter notebooks, hugo_jupyter and some tweaking. Comparison to\
                           R and blogdown'
                          , thumbnailImagePosition = 'left'
                          , thumbnailImage = 'https://gitlab.eurecom.fr/zoe-apps/pytorch/avatar'
                         )


    notebook_to_markdown( path = r'../notebooks/01_r2py_ide.ipynb'
                          , date = '2018-08-21'
                          , slug = 'r2py_ide'
                          , title = 'Moving from R to python - 1/8 - IDE'
                          , author = 'Bjoern Koneswarakantha'
                          , categories = ['R vs. python','IDE']
                          , tags = ['R vs. python','IDE', 'python', 'R', 'jupyter', 'pycharm', 'RStudio']
                          , summary = 'Some reflections on the choice of the python IDE. We end up comparing RStudio to pycharm.'
                          , thumbnailImagePosition = 'left'
                          , thumbnailImage = "r2py.png"
                         )

    notebook_to_markdown( path = r'../notebooks/02_r2py_pandas.ipynb'
                          , date = '2018-08-22'
                          , slug = 'r2py_pandas'
                          , title = 'Moving from R to python - 2/8 - pandas'
                          , author = 'Bjoern Koneswarakantha'
                          , categories = ['R vs. python','pandas']
                          , tags = ['R vs. python','pandas', 'python', 'R', 'dplyr']
                          , summary = 'We look at pandas and compare it to dplyr.'
                          , thumbnailImagePosition = 'left'
                          , thumbnailImage = "r2py.png"
                         )

    notebook_to_markdown( path = r'../notebooks/03_R2Py_matplotlib_seaborn.ipynb'
                          , date = '2018-08-23'
                          , slug = 'r2py_matplotlib_seaborn'
                          , title = 'Moving from R to python - 3/8 - matplotlib and seaborn'
                          , author = 'Bjoern Koneswarakantha'
                          , categories = ['R vs. python','matplotlib', 'seaborn', 'ggplot2']
                          , tags = ['R vs. python','matplotlib', 'seaborn', 'R', 'ggplot2']
                          , summary = 'We look at the visualisations options in python with matplotlib and seaborn.'
                          , thumbnailImagePosition = 'left'
                          , thumbnailImage = "r2py.png"
                         )

    notebook_to_markdown( path = r'../notebooks/04_R2Py_plotly.ipynb'
                          , date = '2018-08-24'
                          , slug = 'r2py_plotly'
                          , title = 'Moving from R to python - 4/8 - plotly'
                          , author = 'Bjoern Koneswarakantha'
                          , categories = ['R vs. python','matplotlib', 'plotly', 'seaborn']
                          , tags = ['R vs. python','plotly', 'R', 'python', 'plotly', 'seaborn']
                          , summary = 'We look at the plotly API for R and python'
                          , thumbnailImagePosition = 'left'
                          , thumbnailImage = "r2py.png"
                         )

    notebook_to_markdown( path = r'../notebooks/05_R2Py_scikitlearn.ipynb'
                          , date = '2018-08-25'
                          , slug = 'r2py_scikitlearn'
                          , title = 'Moving from R to python - 5/8 - scikitlearn'
                          , author = 'Bjoern Koneswarakantha'
                          , categories = ['R vs. python','scikitlearn', 'randomized parameter search']
                          , tags = ['R vs. python','scikitlearn', 'randomized parameter search'
                                    , 'Categorical Encoding', 'matplotlib color maps']
                          , summary = 'We take scikitlearn for a spin, and try out the whole modelling workflow.'
                          , thumbnailImagePosition = 'left'
                          , thumbnailImage = "r2py.png"
                         )

    notebook_to_markdown( path = r'../notebooks/06_R2Py_scikitlearn_advanced.ipynb'
                          , date = '2018-08-26'
                          , slug = 'r2py_scikitlearn_advanced'
                          , title = 'Moving from R to python - 6/8 - scikitlearn'
                          , author = 'Bjoern Koneswarakantha'
                          , categories = ['R vs. python','scikitlearn', 'sklearn-pandas', 'pipes', 'sparse data']
                          , tags = ['R vs. python','scikitlearn', 'randomized parameter search'
                                    , 'sklearn-pandas', 'pipes', 'sparse data']
                          , summary = 'We look into some techniques for scikitlearn that allow us to write more \
                                       generalizable code that executes faster and helps us to avoid numpy arrays.'
                          , thumbnailImagePosition = 'left'
                          , thumbnailImage = "r2py.png"
                         )

    notebook_to_markdown( path = r'../notebooks/07_R2Py_automated_ML.ipynb'
                          , date = '2018-08-27'
                          , slug = 'r2py_automated_ML'
                          , title = 'Moving from R to python - 7/8 - automated machine learning'
                          , author = 'Bjoern Koneswarakantha'
                          , categories = ['R vs. python','tpot', 'auto-sklearn', 'scikitlearn']
                          , tags = ['R vs. python','tpot', 'auto-sklearn', 'scikitlearn']
                          , summary = 'We look into some techniques for scikitlearn that allow us to write more \
                                       generalizable code that executes faster and helps us to avoid numpy arrays.'
                          , thumbnailImagePosition = 'left'
                          , thumbnailImage = "r2py.png"
                         )