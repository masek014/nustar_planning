\LoadClass{article}
\RequirePackage{xparse} % For variable number of command args
\RequirePackage[explicit]{titlesec}
\RequirePackage{graphicx}
\RequirePackage{fancyhdr}
\RequirePackage{microtype}
\RequirePackage[margin=1in]{geometry}


% Modify the section.
\titlespacing*{\section}{0pt}{0\baselineskip}{\baselineskip}
% \titleformat{\section}{\large\normalfont\bfseries}{}{0em}{#1\ \thesection} % Put the section number after the section
% \newcommand\target{\section{Target}}


% This command creates the document title.
\newcommand{\makeplanningtitle}[1]{
    \fancyhf{}
    \fancyhead[C]{\Large #1 \\ \normalsize Compiled \today}
    \fancyfoot[C]{\thepage}
    \thispagestyle{fancy}
}


% This command sets up the header for a new observation.
\newcommand{\addorbit}[2]{
%   \centering
  \Large
  \underline{\bf{#1}: {#2}}
  \vspace{0.5cm}
  \normalsize
  \\
}


% This formats a single target.
% #1  : The target date and number. Ex: Nov 17th Target 1
% #2  : NuSTAR center coordinate.
% #3  : Image file.
% #4+ : Optional target comments.
\DeclareDocumentCommand{\addtarget}{ m m m o o o o o }{%
    \large \textbf{#1} \normalsize \newline
    Time: #2 \newline
    FOV Center: #3 \newline
    \IfValueT{#5}{
        Comments:
        \begin{itemize}
            \item #5 \\
            \IfValueT{#6}{
                \item #6 \\
            }
            \IfValueT{#7}{
                \item #7 \\
            }
            \IfValueT{#8}{
                \item #8 \\
            }
        \end{itemize}
    }
    
    \IfValueT{#4}{
        \begin{figure}[h!]
            \begin{center}
                \includegraphics[width=\textwidth]{#4}
            \end{center}
        \end{figure}
    }
}
