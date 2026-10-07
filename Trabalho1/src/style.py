"""Estilo comum das figuras (impressão em artigo SBC, fonte serifada)."""
import matplotlib as mpl

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GRAY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
MID = "#f0efec"  # ponto médio neutro da escala divergente

LABELS_PT = {
    "Hepatobiliary disorders": "Hepatobiliares",
    "Metabolism and nutrition disorders": "Metabolismo e nutrição",
    "Product issues": "Problemas de produto",
    "Eye disorders": "Oculares",
    "Investigations": "Investigações (exames)",
    "Musculoskeletal and connective tissue disorders": "Musculoesqueléticas",
    "Gastrointestinal disorders": "Gastrointestinais",
    "Social circumstances": "Circunstâncias sociais",
    "Immune system disorders": "Sistema imune",
    "Reproductive system and breast disorders": "Reprodutivas e mama",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)": "Neoplasias",
    "General disorders and administration site conditions": "Gerais e local de admin.",
    "Endocrine disorders": "Endócrinas",
    "Surgical and medical procedures": "Procedimentos médicos",
    "Vascular disorders": "Vasculares",
    "Blood and lymphatic system disorders": "Sangue e linfáticas",
    "Skin and subcutaneous tissue disorders": "Pele e subcutâneo",
    "Congenital, familial and genetic disorders": "Congênitas e genéticas",
    "Infections and infestations": "Infecções e infestações",
    "Respiratory, thoracic and mediastinal disorders": "Respiratórias e torácicas",
    "Psychiatric disorders": "Psiquiátricas",
    "Renal and urinary disorders": "Renais e urinárias",
    "Pregnancy, puerperium and perinatal conditions": "Gravidez e perinatais",
    "Ear and labyrinth disorders": "Ouvido e labirinto",
    "Cardiac disorders": "Cardíacas",
    "Nervous system disorders": "Sistema nervoso",
    "Injury, poisoning and procedural complications": "Lesões e intoxicações",
}


def setup():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": "#b8b7b2",
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def comma_axes(*axes, which="both"):
    """Usa vírgula como separador decimal nos rótulos dos eixos (padrão pt-BR)."""
    from matplotlib.ticker import FuncFormatter

    def f(x, pos=None):
        s = f"{x:g}"
        return s.replace(".", ",").replace("-", "\u2212")
    for ax in axes:
        if which in ("both", "x") and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(FuncFormatter(f))
        if which in ("both", "y") and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(FuncFormatter(f))
