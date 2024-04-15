import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from marvin.tools import Maps
import marvin
import pandas as pd
import copy
import marvin.utils.plot.colorbar as colorbar
from marvin.utils.datamodel.dap import datamodel
from marvin import config
from astropy import units
import matplotlib.patches as mpatches

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
    marvin.config.mode = 'local'
    np.seterr(divide='ignore')
    try: 
        galaxy = marvin.tools.Maps(galaxy_id, bintype='HYB10')
    except:
        raise Exception("Invalid galaxy: " + galaxy_id)
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

    masks, fig, axes = galaxy.get_bpt(use_oi = False, show_plot=False, return_figure=False)

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
    df = pd.DataFrame()
    for galaxy in galaxy_list:
        try:
            counts = get_histogram(galaxy)
            df[galaxy] = counts
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
        except:
            pass

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
    
    df.to_csv('out.csv', index=False) 
    plt.show()
    return df

def plot_PSB(galaxy_id): #Plots PSB layered with Seyfert spaxels and counts the average distance
    marvin.config.mode = 'local'
    try: 
        galaxy = marvin.tools.Maps(galaxy_id, bintype='HYB10')
    except:
        raise Exception("Invalid galaxy: " + galaxy_id)
    alpha = galaxy.emline_gew_ha_6564
    delta = galaxy.specindex_hdeltaagalaxy
    snr = galaxy.spx_snr
    snr_value = getattr(snr, 'value', None)
    
    value = getattr(alpha, 'value', None)
    dvalue = getattr(delta, 'value', None)
    divar = getattr(delta, 'ivar', None)

    imshow_kws = {}
    cb_kws = {}

    masks_bpt = galaxy.get_bpt(use_oi = False, show_plot=False, return_figure=False)

    #print(sum(masks_bpt['seyfert']['sii'] ))
    
    maskpsb = mask_PSB(value, dvalue, divar, snr_value)
    print(maskpsb.sum())
    sf_mask = maskpsb & masks_bpt['seyfert']['sii'] 

    dapmap = alpha
    
    mask = getattr(dapmap, 'mask', np.zeros(value.shape, dtype=bool))
    nocov = _mask_nocov(dapmap, mask, divar)
    low_snr = mask_low_snr(value, divar, 0)

    psb_spax = np.ma.array(np.ones(value.shape) * 100, mask=~maskpsb)
    seyf_spax = np.ma.array(np.ones(value.shape) * 100, mask=~masks_bpt['seyfert']['sii'] )
    comb_spax = np.ma.array(np.ones(value.shape) * 100, mask=~sf_mask)
    #psb_spax = np.ma.array(value, mask=~sf_mask)
    good_spax = np.ma.array(value, mask=np.logical_or.reduce((nocov, low_snr)))

    prop = dapmap.datamodel.full()
    dapver = dapmap._datamodel.parent.release if dapmap is not None else config.lookUpVersions()[1]
    params = datamodel[dapver].get_plot_params(prop)
    cmap = params['cmap']
    percentile_clip = params['percentile_clip']
    symmetric = params['symmetric']
    snr_min = params['snr_min']
    
    cb_kws['cmap'] = cmap
    cb_kws['percentile_clip'] = percentile_clip
    cb_kws['cbrange'] = None
    cb_kws['symmetric'] = symmetric
    cblabel = None

    cblabel = cblabel if cblabel is not None else getattr(dapmap, 'unit', '')
    if isinstance(cblabel, units.UnitBase):
        cb_kws['label'] = cblabel.to_string('latex_inline')
    else:
        cb_kws['label'] = cblabel

    cb_kws['log_cb'] = False
    cb_kws = colorbar._set_cb_kws(cb_kws)
    cb_kws = colorbar._set_cbrange(good_spax, cb_kws)

    extent = _set_extent(value.shape)

    imshow_kws.setdefault('extent', extent)
    imshow_kws.setdefault('interpolation', 'nearest')
    imshow_kws.setdefault('origin', 'lower')
    imshow_kws['norm'] = None
                         
    nocov_kws = copy.deepcopy(imshow_kws)
    nocov_image = np.ma.array(np.ones(value.shape), mask=~nocov.astype(bool))
    A8A8A8 = colorbar._one_color_cmap(color='#A8A8A8')
    
    patch_kws = _set_patch_style({}, extent)
    imshow_kws = colorbar._set_vmin_vmax(imshow_kws, cb_kws['cbrange'])
    
    fig, ax = _ax_setup(False, None, None)

    # plot hatched regions by putting one large patch as lowest layer
    # hatched regions are bad data, low SNR, or negative values if the colorbar is logarithmic
    ax.add_patch(mpl.patches.Rectangle(**patch_kws))

    # plot regions without IFU coverage as a solid color (gray #A8A8A8)
    ax.imshow(nocov_image, cmap=A8A8A8, zorder=1, **nocov_kws)

    # plot unmasked spaxels
    #print(psb_
    #print(psb_spax)
    ax.imshow(psb_spax, cmap='cool', zorder=12, **imshow_kws, alpha=0.7)
    ax.imshow(seyf_spax, cmap='autumn', zorder=12, **imshow_kws, alpha=0.7)
    ax.imshow(comb_spax, cmap='Reds', zorder=12, **imshow_kws, alpha=1)
    p = ax.imshow(good_spax, cmap=cb_kws['cmap'], zorder=10, **imshow_kws)

    cmaps = {1:'yellow',2:'magenta',3:'maroon'}
    labels = {1:'seyfert',2:'psb',3:'both'}
    patches =[mpatches.Patch(color=cmaps[i],label=labels[i]) for i in cmaps]
    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    
    fig, cb = colorbar._draw_colorbar(fig, mappable=p, ax=ax, **cb_kws)

    ax.set_title("Seyfert + PSB overlap for: " + galaxy_id)

    radii = galaxy.spx_ellcoo_elliptical_radius
    radii_value = getattr(radii, 'value', None)
    
    comb_dist = calc_dist(sf_mask, radii_value)
    seyf_dist = calc_dist(masks_bpt['seyfert']['sii'], radii_value)
    psb_dist = calc_dist(maskpsb, radii_value)

    print("Average distance of spaxel with both properties: " + str(np.median(comb_dist)))
    print("Average distance of Seyfert spaxel: " + str(np.median(seyf_dist)))
    print("Average distance of PSB spaxel: " + str(np.median(psb_dist)))

def _mask_nocov(dapmap, mask, ivar=None):

    assert (dapmap is not None) or (mask is not None) or (ivar is not None)

    if dapmap is None:
        pixmask = Maskbit('MANGA_DAPPIXMASK')
        pixmask.mask = mask
    else:
        pixmask = dapmap.pixmask

    try:
        return pixmask.get_mask('NOCOV')
    except (MarvinError, AttributeError, IndexError, TypeError):
        return ivar == 0

def _set_patch_style(patch_kws, extent):
    patch_kws_default = dict(xy=(extent[0] + 0.01, extent[2] + 0.01),
                             width=extent[1] - extent[0] - 0.02,
                             height=extent[3] - extent[2] - 0.02, hatch='xxxx', linewidth=0,
                             fill=True, facecolor='#A8A8A8', edgecolor='w', zorder=0)

    for k, v in patch_kws_default.items():
        if k not in patch_kws:
            patch_kws[k] = v

    return patch_kws

def _set_extent(cube_size):
    extent = np.array([0, cube_size[0] - 1, 0, cube_size[1] - 1])

    return extent

def _ax_setup(sky_coords, fig=None, ax=None, facecolor='#A8A8A8'):
    xlabel = 'arcsec' if sky_coords else 'spaxel'
    ylabel = 'arcsec' if sky_coords else 'spaxel'

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if int(mpl.__version__.split('.')[0]) <= 1:
        ax.set_axis_bgcolor(facecolor)
    else:
        ax.set_facecolor(facecolor)
        ax.grid(False, which='both', axis='both')

    return fig, ax

def calc_dist(mask, radii):
    valid_spax = np.ma.array(radii, mask=~mask)
    data = valid_spax[valid_spax.mask == False]
    #print(data)
    return data
    
def median_seyf_distance(galaxy_id): #computes median of distances from center of galaxy to a seyfert spaxel
    try: 
        galaxy = marvin.tools.Maps(galaxy_id, bintype='HYB10')
    except:
        raise Exception("Invalid galaxy: " + galaxy_id)
        
    radii = galaxy.spx_ellcoo_r_h_kpc
    radii_value = getattr(radii, 'value', None)
    #print(radii)
    #plt.imshow(radii_value, interpolation='none')
    #plt.show()

    masks_bpt = galaxy.get_bpt(use_oi = False, show_plot=False, return_figure=False)
    
    seyf_dist = calc_dist(masks_bpt['seyfert']['sii'], radii_value)

    seyf_dist = seyf_dist.data / (1 / 0.7) #divide by 1 / h where h = 0.7
    #print("Median distance of Seyfert spaxel: " + str(np.median(seyf_dist)))
    #try: 
      #  dap = galaxy.dapall
      #  galaxy_dist = dap['adist_z']
      #  return np.sin(np.median(seyf_dist)*(4.84814 * 10**(-6))) * galaxy_dist
    #except:
      #  raise Exception("Invalid galaxy: " + galaxy_id)
        # return ? 
    if seyf_dist.size < 5: #min 5
        return np.nan
    return np.median(seyf_dist)
    
def plot_seyf_distance(galaxies): 
    dists = []
    non_seyferts = 0
    count = 0
    for i in galaxies:
        if count % 20 == 0:
            print(count)
        median_dist = median_seyf_distance(i)
        if np.isnan(median_dist):
            non_seyferts += 1
        else:
            if median_dist > 3:
                print(i)
            dists.append(median_dist)
        count =+ 1
    n_bins = 15
    print("Galaxies without seyfert region: " + str(non_seyferts))
    plt.hist(dists, bins=n_bins)
    plt.title("Average Distance of Seyfert region to Galactic Center")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()

def psb_LINER_helper(galaxy_id): #Finds average distance of LINER and Seyf spaxel to center for specific galaxy
    try: 
        galaxy = marvin.tools.Maps(galaxy_id, bintype='HYB10')
    except:
        print("Invalid galaxy: " + galaxy_id)
        raise Exception("Invalid galaxy: " + galaxy_id)
    #print(galaxy)
    alpha = galaxy.emline_gew_ha_6564
    delta = galaxy.specindex_hdeltaagalaxy
    snr = galaxy.spx_snr
    snr_value = getattr(snr, 'value', None)
    
    value = getattr(alpha, 'value', None)
    dvalue = getattr(delta, 'value', None)
    divar = getattr(delta, 'ivar', None)

    masks_bpt = galaxy.get_bpt(use_oi = False, show_plot=True, return_figure=False)
    
    maskpsb = mask_PSB(value, dvalue, divar, snr_value)
    radii = galaxy.spx_ellcoo_r_h_kpc
    radii_value = getattr(radii, 'value', None)
    
    liner_mask = maskpsb & masks_bpt['liner']['sii']
    liner_count = np.count_nonzero(masks_bpt['liner']['sii'])
    seyf_count = np.count_nonzero(masks_bpt['seyfert']['sii'])
    
    sf_mask = maskpsb & masks_bpt['seyfert']['sii']
    
    seyf_dist = calc_dist(masks_bpt['seyfert']['sii'], radii_value)
    combseyf_dist = calc_dist(sf_mask, radii_value)

    liner_dist = calc_dist(masks_bpt['liner']['sii'], radii_value)
    combliner_dist = calc_dist(liner_mask, radii_value)

    if liner_dist.size < 5: #min 5
        liner_dist = np.nan
    else:
        liner_dist = liner_dist.data / (1 / 0.7)
    if combliner_dist.size < 5:
        combliner_dist = np.nan
    else:
        combliner_dist = combliner_dist.data / (1 / 0.7)

    if seyf_dist.size < 5: #min 5
        seyf_dist = np.nan
    else:
        seyf_dist = seyf_dist.data / (1 / 0.7)
    if combseyf_dist.size < 5:
        combseyf_dist = np.nan
    else:
        combseyf_dist = combseyf_dist.data / (1 / 0.7)

    print(seyf_count)
    return np.median(liner_dist), np.median(seyf_dist), np.median(combliner_dist), np.median(combseyf_dist), liner_count, seyf_count

def process_LINERpsb(galaxies, filename): #Process large amounts of galaxies to create 2 1-d arrays with average distance of LINER and Seyfert spaxels to center of galaxy
    psbspaxels, nonpsbspaxels = [], []
    count = 0
    df = pd.DataFrame(columns=['IFU', 'LINER_dist', 'Seyf_dist', 'LINERPSB_dist', 'SeyfPSB_dist', 'Liner Count', 'Seyf Count'])
    for i in galaxies: #fix how try and except is handled 
        if count % 20 == 0:
            print(count)
        try:
            LINER, seyf, linercomb, seyfcomb, linercount, seyfcount = psb_LINER_helper(i)
        except:
            print("Invalid galaxy: " + str(i))
            LINER, seyf, linercomb, seyfcomb, linercount, seyfcount = -1, -1, -1, -1, -1, -1
            pass
        df.loc[count] = [i] + [LINER] + [seyf] + [linercomb] + [seyfcomb] + [linercount] + [seyfcount]
        count += 1
    df.to_csv(filename + '.csv', index=False)
    
def comparison_plot_dist(psbgalaxies, nonpsbgalaxies): #Helper function that takes pre-determined distance arrays and creates histogram. For Seyfert distances.
    psbgalaxy_psbspaxels, psbgalaxy_nonpsbspaxels = psbgalaxies
    galaxy_psbspaxels, galaxy_nonpsbspaxels = nonpsbgalaxies
    
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    n_bins = 15
    axs[0].hist([psbgalaxy_psbspaxels, galaxy_psbspaxels], bins='auto', label=['PSB Galaxy', 'Non-PSB Galaxy'], density=True, histtype='step', cumulative=False)
    axs[0].set_title("PSB and Seyfert")
    #w = axs[0].get_yticks() 
    #axs[0].set_yticks(w,np.round(w/(len(psbgalaxy_psbspaxels) + len(galaxy_psbspaxels)),3))
    axs[0].set_xticks(np.arange(0, 3.5, 0.5))
    axs[1].hist([psbgalaxy_nonpsbspaxels, galaxy_nonpsbspaxels], bins='auto', label=['PSB Galaxy', 'Non-PSB Galaxy'], density=True, histtype='step', cumulative=False)
    #axs[1].set_xticks(np.arange(0, 25, 1))
    axs[1].set_title("Seyfert")
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    plt.show()

def comparison_plot_dist_LINER(psbgalaxies, nonpsbgalaxies): #Helper function that takes pre-determined distance arrays and creates histogram. For LINER distances.
    psbgalaxy_psbspaxels, psbgalaxy_nonpsbspaxels = psbgalaxies
    galaxy_psbspaxels, galaxy_nonpsbspaxels = nonpsbgalaxies
    
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    n_bins = 15
    axs[0].hist([psbgalaxy_psbspaxels, galaxy_psbspaxels], bins='auto', label=['PSB Galaxy', 'Non-PSB Galaxy'], density=True, histtype='step', cumulative=True)
    axs[0].set_title("PSB and LINER")
    #w = axs[0].get_yticks() 
    #axs[0].set_yticks(w,np.round(w/(len(psbgalaxy_psbspaxels) + len(galaxy_psbspaxels)),3))
    axs[0].set_xticks(np.arange(0, 3.5, 0.5))
    axs[1].hist([psbgalaxy_nonpsbspaxels, galaxy_nonpsbspaxels], bins='auto', label=['PSB Galaxy', 'Non-PSB Galaxy'], density=True, histtype='step', cumulative=True)
    #axs[1].set_xticks(np.arange(0, 25, 1))
    axs[1].set_title("LINER")
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    plt.show()

def plot_spaxel_frequency(allgalaxies_spax):  #Helper function, takes 1-d array with amount of LINER or Seyfert spaxels for a galaxy and plots it as a histogram.
    plt.hist(allgalaxies_spax, label=['spaxel'], density=False, histtype='step')
    plt.title("Amount of Spaxels in Galaxy")
    plt.xlabel("Amount of Spx")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    #plt.xlim(0,50)
    plt.show()

def visualize_seyf(galaxy_id):   # Returns plot of H-Alpha with Seyfert, PSB areas as a highlighted overlay.
    galaxy = marvin.tools.Maps(galaxy_id, bintype='HYB10')
    alpha = galaxy.emline_gew_ha_6564
    delta = galaxy.specindex_hdeltaagalaxy
    snr = galaxy.spx_snr
    snr_value = getattr(snr, 'value', None)
    
    value = getattr(alpha, 'value', None)
    dvalue = getattr(delta, 'value', None)
    divar = getattr(delta, 'ivar', None)

    imshow_kws = {}
    cb_kws = {}

    masks_bpt = galaxy.get_bpt(use_oi = False, show_plot=True, return_figure=False)

    #print(sum(masks_bpt['seyfert']['sii'] ))
    
    maskpsb = mask_PSB(value, dvalue, divar, snr_value)
    print(maskpsb.sum())
    sf_mask = maskpsb & masks_bpt['seyfert']['sii'] 

    dapmap = alpha
    
    mask = getattr(dapmap, 'mask', np.zeros(value.shape, dtype=bool))
    nocov = _mask_nocov(dapmap, mask, divar)
    low_snr = mask_low_snr(value, divar, 0)

    psb_spax = np.ma.array(np.ones(value.shape) * 100, mask=~maskpsb)
    seyf_spax = np.ma.array(np.ones(value.shape) * 100, mask=~masks_bpt['seyfert']['sii'] )
    comb_spax = np.ma.array(np.ones(value.shape) * 100, mask=~sf_mask)
    #psb_spax = np.ma.array(value, mask=~sf_mask)
    good_spax = np.ma.array(value, mask=np.logical_or.reduce((nocov, low_snr)))

    prop = dapmap.datamodel.full()
    dapver = dapmap._datamodel.parent.release if dapmap is not None else config.lookUpVersions()[1]
    params = datamodel[dapver].get_plot_params(prop)
    cmap = params['cmap']
    percentile_clip = params['percentile_clip']
    symmetric = params['symmetric']
    snr_min = params['snr_min']
    
    cb_kws['cmap'] = cmap
    cb_kws['percentile_clip'] = percentile_clip
    cb_kws['cbrange'] = None
    cb_kws['symmetric'] = symmetric
    cblabel = None

    cblabel = cblabel if cblabel is not None else getattr(dapmap, 'unit', '')
    if isinstance(cblabel, units.UnitBase):
        cb_kws['label'] = cblabel.to_string('latex_inline')
    else:
        cb_kws['label'] = cblabel

    cb_kws['log_cb'] = False
    cb_kws = colorbar._set_cb_kws(cb_kws)
    cb_kws = colorbar._set_cbrange(good_spax, cb_kws)

    extent = _set_extent(value.shape)

    imshow_kws.setdefault('extent', extent)
    imshow_kws.setdefault('interpolation', 'nearest')
    imshow_kws.setdefault('origin', 'lower')
    imshow_kws['norm'] = None
                         
    nocov_kws = copy.deepcopy(imshow_kws)
    nocov_image = np.ma.array(np.ones(value.shape), mask=~nocov.astype(bool))
    A8A8A8 = colorbar._one_color_cmap(color='#A8A8A8')
    
    patch_kws = _set_patch_style({}, extent)
    imshow_kws = colorbar._set_vmin_vmax(imshow_kws, cb_kws['cbrange'])
    
    fig, ax = _ax_setup(False, None, None)

    # plot hatched regions by putting one large patch as lowest layer
    # hatched regions are bad data, low SNR, or negative values if the colorbar is logarithmic
    ax.add_patch(mpl.patches.Rectangle(**patch_kws))

    # plot regions without IFU coverage as a solid color (gray #A8A8A8)
    ax.imshow(nocov_image, cmap=A8A8A8, zorder=1, **nocov_kws)

    # plot unmasked spaxels
    #print(psb_
    #print(psb_spax)
    ax.imshow(psb_spax, cmap='cool', zorder=12, **imshow_kws, alpha=0.7)
    ax.imshow(seyf_spax, cmap='autumn', zorder=12, **imshow_kws, alpha=0.7)
    ax.imshow(comb_spax, cmap='Reds', zorder=12, **imshow_kws, alpha=1)
    p = ax.imshow(good_spax, cmap=cb_kws['cmap'], zorder=10, **imshow_kws)

    cmaps = {1:'yellow',2:'magenta',3:'maroon'}
    labels = {1:'seyfert',2:'psb',3:'both'}
    patches =[mpatches.Patch(color=cmaps[i],label=labels[i]) for i in cmaps]
    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    
    fig, cb = colorbar._draw_colorbar(fig, mappable=p, ax=ax, **cb_kws)

    ax.set_title("Seyfert + PSB overlap for: " + galaxy_id)
        
        
    
