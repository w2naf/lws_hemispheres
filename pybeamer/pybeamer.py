import os
import shutil
import datetime
import subprocess

path            = os.path.abspath(__file__)
this_mod_dir    = os.path.dirname(path)

header = r"""
\documentclass[9pt]{beamer}
\mode<presentation>
{
  	\usetheme{vtech}
	\usecolortheme{}
	\setbeamercovered{transparent}
	\setbeamertemplate{navigation symbols}{}
}
\usepackage[english]{babel}
\usepackage[latin1]{inputenc}
%\usepackage{pgf}
\usepackage{times}
\usepackage[T1]{fontenc}
\usepackage{natbib}
%\\bibpunct{[}{]}{;}{a}{,}{,~}
\usepackage[3D]{sty/movie15}
\usepackage{sty/flow}
\usepackage{sty/eqlist}
\usepackage{url}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{exscale}
\usepackage[mathscr]{eucal}
\usepackage{wasysym} % For double closed integrals
\usepackage{bm}
\usepackage{sty/eqlist} % Makes for a nice list of symbols.
\usepackage{graphicx}
\usepackage{color}
\usepackage{epstopdf}
\usepackage{hyperref}
\hypersetup{
    bookmarks=true,         % show bookmarks bar?
    unicode=false,          % non-Latin characters in acrobat bookmarks
    pdftoolbar=true,        % show acrobat toolbar?
    pdfmenubar=true,        % show acrobat menu?
    pdffitwindow=true,      % page fit to window when opened
    pdftitle={Group Meeeting},    % title
    pdfauthor={N. A. Frissell},     % author
    pdfsubject={Group meeting presentation},   % subject of the document
    pdfnewwindow=true,      % links in new window
    pdfkeywords={MSTID, SuperDARN}, % list of keywords
%    colorlinks=true,       % false: boxed links; true: colored links
%    linkcolor=white,          % color of internal links
%    citecolor=blue,        % color of links to bibliography
%    filecolor=magenta,      % color of file links
%    urlcolor=cyan           % color of external links
}

\usepackage{marvosym}
\usepackage{pifont}
\usepackage{colortbl}
\usepackage{hyperref}
\definecolor{VTgray}{RGB}{194,193,186}
\definecolor{VTdarkgray}{RGB}{64,64,57}
\definecolor{VTmaroon}{RGB}{110, 73, 144}
\definecolor{VTorange}{RGB}{110, 73, 144}
"""

author_list = []
aa  = author_list.append
aa('N.A. Frissell')
all_authors = ', '.join(author_list)

def clean_beamer_path(fpath):
    path,fname  = os.path.split(fpath)

    bn, ext     = os.path.splitext(fname)
    new_fname   = '{'+bn+'}'+ext
    new_fpath   = os.path.join(path,new_fname)
    return new_fpath

class Beamer(object):
    def __init__(self,output_dir='beamer',filename='pres_latex.tex',
            full_title      = 'Full Title',
            running_title   = 'Running Title',
            subtitle        = 'Subtitle',
            subject         = 'Subject',
            first_author    = 'N.A. Frissell',
            all_authors     = all_authors,
            institute       = 'Space@VT',
            date            = None):

        self.output_dir = output_dir
        self.figure_dir = 'Figures'
        self.filename   = filename
        self.filepath   = os.path.join(self.output_dir,self.filename)

        try:
            shutil.rmtree(output_dir)
        except:
            pass

        shutil.copytree(os.path.join(this_mod_dir,'beamer'),output_dir)

        self.latex  = header.split('\n')
        lx = self.latex.append
        
        #### Title
        lx('\\title[{!s}]'.format(running_title))
        lx('{{{!s}}}'.format(full_title))
        lx('')

        #### Subtitle
        lx('%\\subtitle[{!s}]'.format(subtitle))
        lx('%{{{!s}}}'.format(subtitle))
        lx('')

        #### Author
        lx('\\author[{!s}]'.format(first_author))
        lx('{{{!s}}}'.format(all_authors))
        lx('')

        #### Institute
        lx('\\institute[{!s}]'.format(institute))
        lx('{{{!s}}}'.format(institute))
        lx('')

        #### Date
        date = datetime.datetime.now().strftime('%d %B %Y')
        lx('\\date[{!s}]'.format(date))
        lx('{{\\small {!s}}}'.format(date))
        lx('')

        #### Subject
        lx('\\subject{{{!s}}}'.format(subject))

        #### University Logo
        txt = """
\pgfdeclareimage[height=0.5cm]{university-logo}{Figures/vt_logo.pdf}

\AtBeginSection[]
{
  \\begin{frame}<beamer>{Outline}
    \\tableofcontents[currentsection,hideothersubsections]
  \\end{frame}
}

\\begin{document}

\\frame[plain]{
   \\titlepage 
}

\\begin{frame}{Outline}
    \\tableofcontents[hideallsubsections]
\end{frame}"""
        lx(txt)
        lx('')

    def add_section(self,section):
        if section is not None:
            section = section.replace('_','\_')
        txt = '\\section[{0}]{{{0}}}'.format(str(section))
        self.latex.append(txt)

    def add_subsection(self,subsection):
        if subsection is not None:
            subsection = subsection.replace('_','\_')
        txt = '\\subsection[{0}]{{{0}}}'.format(str(subsection))
        self.latex.append(txt)

    def add_fig_slide(self,title,fig_path,basename=None,width='11.5cm'):
        """
        basename: new name of figure file. If None, use the same basename
                    as the original file.
        """
        if basename is None:
            basename        = os.path.basename(fig_path)

        beamer_fig_path = os.path.join(self.figure_dir,basename)
        cp_fig_path     = os.path.join(self.output_dir,beamer_fig_path)
        status          = shutil.copy2(fig_path,cp_fig_path)

        title = title.replace('_','\_')

        lx  = self.latex.append
        lx('\\begin{{frame}}{{{!s}}}'.format(title))
        lx('  \\begin{columns}[c]')
        lx('    \\begin{column}{12cm}')
#        lx('      \\begin{{center}} \\includegraphics[width=11.5cm]{{{!s}}} \\end{{center}}'.format(clean_beamer_path(beamer_fig_path)))
        lx('      \\begin{{center}} \\includegraphics[width={!s}]{{{!s}}} \\end{{center}}'.format(width,beamer_fig_path))
        lx('    \\end{column}')
        lx('  \\end{columns}')
        lx('\\end{frame}')
        lx('')

        return cp_fig_path

    def add_figs_slide(self,title,figs,width=11.5):
        title = title.replace('_','\_')

        lx  = self.latex.append
        lx('\\begin{{frame}}{{{!s}}}'.format(title))
        lx('  \\begin{columns}[c]')
        lx('    \\begin{column}{12cm}')

        for fig_path in figs:
            basename        = os.path.basename(fig_path)
            beamer_fig_path = os.path.join(self.figure_dir,basename)
            cp_fig_path     = os.path.join(self.output_dir,beamer_fig_path)
            status          = shutil.copy2(fig_path,cp_fig_path)
            lx('      \\begin{{center}} \\includegraphics[width={:g}cm]{{{!s}}} \\end{{center}}'.format(width,clean_beamer_path(beamer_fig_path)))

        lx('    \\end{column}')
        lx('  \\end{columns}')
        lx('\\end{frame}')
        lx('')

    def add_figs_slide_cols(self,title,figs,width=11.5,col_width=None):
        if col_width is None:
            col_width = width * 12/11.5
        title = title.replace('_','\_')

        lx  = self.latex.append
        lx('\\begin{{frame}}{{{!s}}}'.format(title))
        lx('  \\begin{columns}[c]')

        for fig_path in figs:
            lx('    \\begin{{column}}{{{:g}cm}}'.format(col_width))
            basename        = os.path.basename(fig_path)
            beamer_fig_path = os.path.join(self.figure_dir,basename)
            cp_fig_path     = os.path.join(self.output_dir,beamer_fig_path)
            status          = shutil.copy2(fig_path,cp_fig_path)
            lx('      \\begin{{center}} \\includegraphics[width={:g}cm]{{{!s}}} \\end{{center}}'.format(width,clean_beamer_path(beamer_fig_path)))

            lx('    \\end{column}')
        lx('  \\end{columns}')
        lx('\\end{frame}')
        lx('')

    def write_latex(self):
        self.latex.append('\\end{document}')
        with open(self.filepath,'w') as fl:
            fl.write('\n'.join(self.latex))

    def make(self):
        cmd = ['make','-C',self.output_dir]
        subprocess.check_call(cmd)
