import numpy as np
import matplotlib.pyplot as plt
from marvin.tools import Maps
import marvin

def mask_PSB(hvalue, dvalue, divar, snr):  #returns a mask of all spaxels that satisfy the PSB criteria, params are 2d arrays, hvalue = halpha, dvalue = hdelta, divar = hdelta inverse variance, snr = signal to noise ratio
    psb = np.zeros(hvalue.shape, dtype=bool)

    var = 1 / np.sqrt(divar)
    psb[np.logical_and(np.logical_and(hvalue < 3, (dvalue - var) > 4), snr > 3)] = True

    return psb

def get_total_spaxels(dapmap): #returns a total amount of all spaxels with scientific data from a galaxy as int
    value = getattr(dapmap, 'value', None)
    ivar = getattr(dapmap, 'ivar', None)
    low_snr = mask_low_snr(value, ivar, 3)
    nocov = dapmap.pixmask.get_mask('NOCOV') 
    good_spax = np.ma.array(value, mask=np.logical_or.reduce((nocov, low_snr))) #This method is most similar to how marvin uses masks 
                                                                                #to cover unwanted spaxels, but there are many ways to do this.
    valid_counts = good_spax.count()
    return valid_counts

def mask_low_snr(value, ivar, snr_min): #marvin function to mask low_snr spaxels to ignore them as valid spaxels, returns a mask where spaxels have too low_snr

    low_snr = np.zeros(value.shape, dtype=bool)

    if (ivar is not None) and (not np.all(np.isnan(ivar))):
        low_snr = (ivar == 0.)

        if snr_min is not None:
            low_snr[np.abs(value * np.sqrt(ivar)) < snr_min] = True

    return low_snr

def get_histogram(galaxy_id): #returns the histogram frequencies for one galaxy, input is its plateifu ID. 
    np.seterr(divide='ignore')
    galaxy = marvin.tools.Maps(galaxy_id, bintype='HYB10')
    alpha = galaxy.emline_gew_ha_6564
    delta = galaxy.specindex_hdeltaagalaxy
    snr = galaxy.spx_snr
    snr_value = getattr(snr, 'value', None)
    
    value = getattr(alpha, 'value', None)
    dvalue = getattr(delta, 'value', None)
    divar = getattr(delta, 'ivar', None)
    
    maskpsb = mask_PSB(value, dvalue, divar, snr_value)
    total_spax = get_total_spaxels(alpha)
    #print(np.count_nonzero(maskpsb))

    masks, fig, axes = galaxy.get_bpt(use_oi = False, show_plot=False)

    comp_mask = maskpsb & masks['comp']['nii'] 
    comp_psbcount = np.count_nonzero(comp_mask)
    comp_nopsb = np.count_nonzero(masks['comp']['nii']) - comp_psbcount

    sf_mask = maskpsb & masks['sf']['nii'] 
    sf_psbcount = np.count_nonzero(sf_mask)
    sf_nopsb = np.count_nonzero(masks['sf']['nii']) - sf_psbcount

    amb_mask = maskpsb & masks['ambiguous']['global'] 
    amb_psbcount = np.count_nonzero(amb_mask)
    amb_nopsb = np.count_nonzero(masks['ambiguous']['global']) - amb_psbcount

    seyf_mask = maskpsb & masks['seyfert']['sii'] 
    seyf_psbcount = np.count_nonzero(seyf_mask)
    seyf_nopsb = np.count_nonzero(masks['seyfert']['sii']) - seyf_psbcount

    liner_mask = maskpsb & masks['liner']['sii'] 
    liner_psbcount = np.count_nonzero(liner_mask)
    liner_nopsb = np.count_nonzero(masks['liner']['sii']) - liner_psbcount

    psb_noncategorized = np.count_nonzero(maskpsb) - comp_psbcount - sf_psbcount - amb_psbcount - seyf_psbcount - liner_psbcount
    nonpsb_noncategorized = np.count_nonzero(masks['invalid']['global'])

    return [comp_psbcount, comp_nopsb, sf_psbcount, sf_nopsb, amb_psbcount, amb_nopsb, seyf_psbcount, seyf_nopsb, liner_psbcount, liner_nopsb, psb_noncategorized, nonpsb_noncategorized]

def plot(galaxy_list): #Creates histogram for a list of plateifus.
    comp_psbTotal = 0.
    comp_nopsbTotal = 0
    sf_psbTotal = 0
    sf_nopsbTotal = 0
    amb_psbTotal = 0
    amb_noTotal = 0
    seyf_psbTotal = 0
    seyf_noTotal = 0
    liner_psbTotal = 0
    liner_nopsbTotal = 0
    psb_noncategorized = 0
    nopsb_noncategorized = 0
    index = 0
    for galaxy in galaxy_list:
        counts = get_histogram(galaxy)
        index += 1
        if index % 20 == 0:
            print(index)
        comp_psbTotal += counts[0]
        comp_nopsbTotal += counts[1]
        sf_psbTotal += counts[2]
        sf_nopsbTotal += counts[3]
        amb_psbTotal += counts[4]
        amb_noTotal += counts[5]
        seyf_psbTotal += counts[6]
        seyf_noTotal += counts[7]
        liner_psbTotal += counts[8]
        liner_nopsbTotal += counts[9]
        psb_noncategorized += counts[10]
        nopsb_noncategorized += counts[11]

    categories = ('Star Forming', 'Composite', 'Ambiguous', 'Seyfert', 'LINER', 'Nonclassified')
    category_counts = {
    'PSB': (sf_psbTotal, comp_psbTotal, amb_psbTotal, seyf_psbTotal, liner_psbTotal, psb_noncategorized),
    'No PSB': (sf_nopsbTotal, comp_nopsbTotal, amb_noTotal, seyf_noTotal, liner_nopsbTotal, nopsb_noncategorized),
    }

    x = np.arange(len(categories))  # the label locations
    width = 0.4
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for category, count in category_counts.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, count, width, label=category)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Frequency')
    ax.set_title('PSB vs non-PSB Spaxels for all galaxies')
    ax.set_xticks(x + width - 0.20, categories)
    ax.legend(loc='upper left')
    maxy = max([sf_psbTotal, comp_psbTotal, amb_psbTotal, seyf_psbTotal, liner_psbTotal, sf_nopsbTotal, comp_nopsbTotal, amb_noTotal, seyf_noTotal, liner_nopsbTotal])
    maxy = 1.2*maxy
    ax.set_ylim([0, maxy])
    ax.text(0.87, 0.76, str(nopsb_noncategorized),
        horizontalalignment='right',
        verticalalignment='top',
        transform = ax.transAxes)

    plt.show()