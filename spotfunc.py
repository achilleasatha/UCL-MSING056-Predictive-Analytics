
# Import libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import collections
import networkx as nx
import seaborn
import warnings
 # Repeat warning everytime
# --------------------------------------------------------------------------------

#PREPARE DATA

# Read all CSVs from specified folder
def read_all_csvs(path):
    os.chdir(path)
    all_csv=[]
    for file in glob.glob("*.csv"):
        all_csv.append(file)
    data_sets={}
    for x in all_csv:
        name = x.replace(" ", "_")
        data_sets[name] = pd.read_csv(x,error_bad_lines=False,warn_bad_lines=False, encoding='latin-1')
    return(data_sets)

def read_all_tsvs(path):
    os.chdir(path)
    all_csv=[]
    for file in glob.glob("*.csv"):
        all_csv.append(file)
    data_sets={}
    for x in all_csv:
        name = x.replace(" ", "_")
        data_sets[name] = pd.read_table(x,encoding = 'latin-1',error_bad_lines=False,warn_bad_lines=False)
    return(data_sets)
# --------------------------------------

# Drop NA values from node and link columns
def clean_data(df,node,link, node_minimum_nchar, link_minimum_nchar):
    df = df.dropna(subset=[node,link])
    df[node] = df[node].map(str)
    df[link] = df[link].map(str)
    df = df[(df[node].map(len)>node_minimum_nchar) & (df[link].map(len)>link_minimum_nchar)]
    return(df)


#Remove Node_Link combinations
def remove_duplicate_node_link(df,node,link):
    df['node_link'] = df[node].map(str)+df[link].map(str)
    df = df.drop_duplicates(subset=['node_link'])
    return(df)


# Links per unique node (feature creation)
# Find number of links per unique node
def links_per_node(df,node,link):
    df=df.pivot_table(link, node,aggfunc=len,)
    s = pd.DataFrame(df.keys())
    t = pd.DataFrame(df.values)
    lpn = pd.concat([s, t], axis=1)
    lpn.columns=[node, 'NumberOfLinks']
    return(lpn)

# Merge links per node width original dataframe
def add_links_per_node(df,node,link):
    return(df.merge(links_per_node(df,node,link),on=node,how='left'))


# Get duplicated users to make sure they link atleast 2 artists
def get_duplicated_links(df,link):
    ids = df[df[link].duplicated()][link].reset_index()
    new_df =  df[df[link].isin(ids[link])]
    return(new_df)


# Get unique artist_x - user - artist_y rows
def get_unique_node_link_node_grouping(df,node,link,node_x,node_y):
    # Get from - to groupings
    new_df = df.merge(df,on = link)
    # Drop duplicate node_x-link-node_y rows
    new_df['ordered_nodes'] = new_df.apply(lambda x: '-'.join(sorted([x[node_x],x[link],x[node_y]])),axis=1)
    new_df = new_df.drop_duplicates(['ordered_nodes'])
    # Drop node_x == node_y rows
    new_df = new_df[new_df[node_x] != new_df[node_y]]
    return(new_df)
# --------------------------------------------------------------------------------
# Aggregate Above functions

def create_network_DATA(df,node,link,node_minimum_nchar,link_minimum_nchar,node_x,node_y):
    # Assign names for node x and node y
    df = clean_data(df,node,link, node_minimum_nchar, link_minimum_nchar)
    df = remove_duplicate_node_link(df,node,link)
    df = add_links_per_node(df,node,link)
    df = get_duplicated_links(df,link)
    df = get_unique_node_link_node_grouping(df,node,link,node_x,node_y)
    return(df)

# Get Nodes
def get_nodes(df,node,node_x,node_y):
    nodes = pd.concat([df[node_x],df[node_y]],ignore_index=True).unique()
    nodes = pd.DataFrame(nodes)
    nodes['id'] = nodes.index
    nodes.columns = [node, 'id']
    return(nodes)


# Get Links and Weights
def get_weights(df,node_x,node_y):
    df = df[[node_x,node_y]]
    df = df.groupby([node_x,node_y]).size().reset_index()
    df.columns=['from','to','weight']
    return(df)

# --------------------------------------------------------------------------------

#PLOT NETWORK

def create_network(df,from_col, to_col, attr):
    import networkx as nx
    G = nx.Graph
    G = nx.from_pandas_dataframe(df,source=from_col,target=to_col,edge_attr=[attr])
    print(nx.info(G))
    return(G)




def plot_network(G,node_size_multiplier,size_font,node_distance):
    plt.figure(figsize=(16,16))
    plt.axis('off')
    # Spring Layout
    layout = nx.spring_layout(G,k=node_distance)
    # Edge thickness as weights
    weights = [G[u][v]['weight']/10000 for u,v in G.edges()]
    # Node size as degree of centrality
    d = nx.degree(G)
    nx.draw_networkx_nodes(G, pos=layout,
                        nodelist=d.keys(),node_size=[x * node_size_multiplier for x in d.values()],
                        alpha=0.8,node_color='orange')
    nx.draw_networkx_edges(G, pos=layout, width=weights,
                        style='solid', edge_color='black')
    nx.draw_networkx_labels(G, pos=layout, font_size=size_font)
    return(plt.show())

# --------------------------------------------------------------------------------

# NETWORK METRICS

def degree(G,node):
    degree = nx.degree(G)
    degree = sorted(degree.items(),key = lambda x: x[1],  reverse=True)
    degree = pd.DataFrame(degree)
    degree.columns = [node,'degree']
    return(degree)


# Calculate Degree Centrality
def degree_centrality(G,node):
    degree_centrality = nx.degree_centrality(G)
    degree_centrality = sorted(degree_centrality.items(),key = lambda x: x[1],  reverse=True)
    degree_centrality = pd.DataFrame(degree_centrality)
    degree_centrality.columns = [node,'degree_centrality']
    return(degree_centrality)



# Calculate Closeness Centrality
def closeness_centrality(G,node):
    closeness_centrality = nx.closeness_centrality(G)
    closeness_centrality = sorted(closeness_centrality.items(),key = lambda x: x[1],  reverse=True)
    closeness_centrality = pd.DataFrame(closeness_centrality)
    closeness_centrality.columns = [node,'closeness_centrality']
    return(closeness_centrality)


# Calculate Betweeness Centrality
def betweeness_centrality(G,node):
    betweeness_centrality = nx.betweenness_centrality(G)
    betweeness_centrality = sorted(betweeness_centrality.items(),key = lambda x: x[1],  reverse=True)
    betweeness_centrality = pd.DataFrame(betweeness_centrality)
    betweeness_centrality.columns = [node,'betweeness_centrality']
    return(betweeness_centrality)

def eigenvector_centrality(G,node):

    eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
    eigenvector_centrality = sorted(eigenvector_centrality.items(),key = lambda x: x[1],  reverse=True)
    eigenvector_centrality = pd.DataFrame(eigenvector_centrality)
    eigenvector_centrality.columns = [node,'eigenvector_centrality']
    return(eigenvector_centrality)

# Aggregate Above Functions

# Compare Centrality Metrics
def compare_centrality_metrics(G,sort_col,node):
    df_compare_metrics = degree(G,node).merge(degree_centrality(G,node),on=node).merge(closeness_centrality(G,node),on=node).merge(betweeness_centrality(G,node),on=node).merge(eigenvector_centrality(G,node),on=node).sort_values([sort_col],ascending=False)
    return(df_compare_metrics)

# --------------------------------------------------------------------------------

# LINK PREDICTION

# Using Jaccard Coefficient
def pred_jc(G,node):
    pred_jc = nx.jaccard_coefficient(G)
    pred_jc_dict = {}
    for u,v,p in pred_jc:
        pred_jc_dict[(u,v)]=p
    pred_jc_dict = sorted(pred_jc_dict.items(),key = lambda x: x[1],  reverse=True)
    pred_jc_dict = pd.DataFrame(pred_jc_dict)
    pred_jc_dict.columns=[node+' link prediction','Jaccard Coefficient']
    return(pred_jc_dict)



# Using Preferential Attachment
def pred_pa(G,node):
    pred_pa= nx.preferential_attachment(G)
    pred_pa_dict = {}
    for u,v,p in pred_pa:
        pred_pa_dict[(u,v)]=p
    pred_pa_dict = sorted(pred_pa_dict.items(),key = lambda x: x[1],  reverse=True)
    pred_pa_dict = pd.DataFrame(pred_pa_dict)
    pred_pa_dict.columns=[node+' link prediction','Preferential Assessment Score']
    return(pred_pa_dict)


################################################################################################

#FEATURED ARTISTS

def add_datetime_detail(data):
    data['DateTime'] = pd.to_datetime(data.log_time)
    data['hour'] = data.DateTime.dt.hour
    data['minute'] = data.DateTime.dt.minute
    data['week'] = data.DateTime.dt.week
    data['month'] = data.DateTime.dt.month
    data['year'] = data.DateTime.dt.year
    data['day'] = data.DateTime.dt.day
    data['date'] = data.DateTime.dt.date
    data['weekday'] = data.DateTime.dt.weekday
    data['weekday_name'] = data.DateTime.dt.weekday_name

# --------------------------------------------------------------------------------


def remove_char(row):
    row.track_artists = row.track_artists.replace('[','')
    row.track_artists = row.track_artists.replace(']','')
    return row

def split_artists(row,keywords):
    both_track_artists = row.lower()
    for i in keywords:
        if (i in both_track_artists) & (both_track_artists.find(i)!=0):
            artists_list = both_track_artists.split(i)
            return [x.strip() for x in artists_list]



# Aggregate above functions

def add_split_artists_columns(df,keywords):
    # df = df.apply(remove_char,axis=1) # Remove '[', or ']' characters
    df = df[df.track_artists.str.lower().str.contains('|'.join(keywords))] #subset featured artists rows
    df['artist_1'] = df.track_artists.apply(lambda row: split_artists(row, keywords)[0])
    df['artist_2'] = df.track_artists.apply(lambda row: split_artists(row, keywords)[1])
    return df

# --------------------------------------------------------------------------------

# Aggregate above functions to get first play

def get_featured_artists_isrc_first_stream(df,select_cols, keywords):

    # Add spaces before and after keywords to ensure they are seperate words and not part of an artist name
    puncs = [':','.',""]
    keywords = [" " + (x+i) + " " for x in keywords for i in puncs]

    if 'DateTime' not in select_cols:
        select_cols.append('DateTime')

    # Keep selected columns
    ndf = df[select_cols]

    #Split track_artists into artist 1 & 2
    ndf = add_split_artists_columns(ndf, keywords)

    # Keep only first appearence (stream) of featured artist pair
    ndf = ndf.sort_values(['DateTime'])
    df_isrc = ndf.drop_duplicates(['isrc'],keep = 'first')

    return df_isrc

# -------------------------------------------------------------
# Create Network Metrics

def create_network_metrics(df,metric,node,link,node_minimum_nchar,link_minimum_nchar,node_x,node_y,sort_col):
    network_data = create_network_DATA(df,node,link,node_minimum_nchar,link_minimum_nchar,node_x,node_y)
    weight_data = get_weights(network_data,node_x,node_y)
    t = create_network(weight_data,'from','to','weight')

    if metric == 'all':
        metric_data = compare_centrality_metrics(t,sort_col,node)
    else:
        metric_data = eval('%s(t,node)'%metric)

    metric_data['artist_name'] = metric_data.artist_name.str.lower()
    return metric_data

def create_network_data_splits(df,split_unit,after_switch = False):
    df_holder = {}
    df.artist_name = df.artist_name.str.lower() #Make lowercase for match
    for i in df[split_unit].unique():
        df_holder["before_{0}_{1}".format(i,split_unit)]= df.loc[df[split_unit] <= i]
        if after_switch:
            df_holder["after_{0}_{1}".format(i,split_unit)]= df.loc[df[split_unit] > i]
    return df_holder

# -------------------------------------------------------------
# Create Final DF

def gender_percentage(df):
    if (len(df)!=0):
        df = df.drop_duplicates(subset = ['customer_id'])
        perc = len(df[df.gender == 'male'])/len(df)
        return perc
    else:
        return 0


def age_percentages(df):
    df = df.dropna(subset = ['birth_year'])
    # df = df[(df.birth_year!='male') & (df.birth_year!='female')] # for required for 2017 data
    df.birth_year = np.float64(df.birth_year)
    if 'DateTime' not in df.columns:
        df['DateTime'] = pd.to_datetime(df.log_time)
    df['age'] = df.year - df.birth_year
    df = df.drop_duplicates(subset = ['customer_id'])
    bins = [0, 18, 25, 50, 100]
    group_names = ['0-18', '19-25', '26-50', '51-100']
    categories = pd.cut(df['age'], bins, labels=group_names)
    return {y:round(x/categories.value_counts().sum(), 2) for x,y in zip(categories.value_counts(),categories.value_counts().keys())}


def apply_query_metrics(df,split_unit_dfs,network_metrics,network_metrics_total,selected_metric,all_data,split_unit):

    def add_query_metrics(row):

        # isrc
        row['isrc_listens'] = len(all_data[all_data.isrc == row.isrc])
        row['isrc_unique_customers'] = len(all_data[all_data.isrc == row.isrc].customer_id.unique())
        row['isrc_percentage_male'] = gender_percentage(all_data[all_data.isrc == row.isrc])
        row['isrc_age_percentage'] = age_percentages(all_data[all_data.isrc == row.isrc])

        group_names = ['Dependent', 'YoungAdult', 'Adult', 'Senior']
        for i in group_names:
            row['isrc_age_perc_{0}'.format(i)] = age_percentages(all_data[all_data.isrc == row.isrc])[i]


        # Get before and after for each artist - using month as filter
        tdate = row[split_unit]
        # tdate = row['month']
        for i in ['artist_1','artist_2']:

            selected_artist = row[i]

            # Select two dataframes for each artist (all months before and all after)

           #months
            b_a = {}
            for p in ['before','after']:
                temp = split_unit_dfs['{0}_{1}_{2}'.format(p,tdate,split_unit)]
                b_a['{0}_dfs_{1}'.format(split_unit,p)] = temp
            b_a['{0}_total'.format(split_unit)]= all_data


            #selected artists
            artist_before = b_a['{0}_dfs_before'.format(split_unit)][b_a['{0}_dfs_before'.format(split_unit)].artist_name == selected_artist]
            artist_after = b_a['{0}_dfs_after'.format(split_unit)][b_a['{0}_dfs_after'.format(split_unit)].artist_name == selected_artist]
            artist_total = b_a['{0}_total'.format(split_unit)][b_a['{0}_total'.format(split_unit)].artist_name == selected_artist]


            # month_dfs_before = month_dfs['before_month_%s'%tdate]
            # artist_before = month_dfs_before[month_dfs_before.artist_name == selected_artist]


           #streams

            row["%s_listens_before_collaboration"%i] = len(artist_before)
            row["%s_listens_after_collaboration"%i] = len(artist_after)
            row["%s_listens_total_uptil_today"%i] = len(artist_total)


            #customers

            row["%s_unique_customer_before_collaboration"%i] = len(artist_before.customer_id.unique())
            row["%s_unique_customer_after_collaboration"%i] = len(artist_after.customer_id.unique())
            row["%s_unique_customer_uptil_today"%i] = len(artist_total.customer_id.unique())

            #customer gender stats

            row["%s_percentage_male_before_collaboration"%i] = gender_percentage(artist_before)
            row["%s_percentage_male_after_collaboration"%i] = gender_percentage(artist_after)
            row["%s_percentage_male_uptil_today"%i] = gender_percentage(artist_total)

            #customer age stats
            group_names = ['Dependent', 'YoungAdult', 'Adult', 'Senior']
            for x in group_names:
                row["{0}_age_percentages_before_collaboration_{1}".format(i,x)] = age_percentages(artist_before)[x]
                row["{0}_age_percentages_after_collaboration_{1}".format(i,x)] = age_percentages(artist_after)[x]
                row["{0}_age_percentages_uptil_today_{1}".format(i,x)] = age_percentages(artist_total)[x]


            # AVG before & after

            # avg_unit = 'week'
            avg_count_over = 'minute'

            # def get_average(df,avg_unit,avg_count_over):
            #     return df.groupby([avg_unit])[avg_count_over].mean()

            def get_average(df,avg_count_over):
                return df.groupby([avg_count_over]).size().mean()

            row["%s_AVG_listens_before_collaboration_%s"%(i,avg_count_over)] = get_average(artist_before,avg_count_over)
            row["%s_AVG_listens_after_collaboration_%s"%(i,avg_count_over)] = get_average(artist_after,avg_count_over)
            row["%s_AVG_listens_uptil_today_%s"%(i,avg_count_over)] = get_average(artist_total,avg_count_over)



            # Network Metrics  - Query only network metrics data


            error_switch = False

            def return_network_metric(df):
                if len(df) == 0:
                    if error_switch:
                        return 'Artist not found in network'
                    else:
                        return None
                elif len(df) >1:
                    if error_switch:
                        return 'Error:'+str(len(df))+' artists found in network'
                    else:
                        return None
                else:
                    return df[selected_metric]

             # total
            artist_total_metrics = network_metrics_total[network_metrics_total.artist_name == selected_artist]
            row['{0}_{1}_uptil_today'.format(i,selected_metric)] = return_network_metric(artist_total_metrics)


            # months before date
            name_before = 'network_metrics_before_{0}_{1}.csv'.format(tdate,split_unit)
            if name_before in network_metrics:
                metrics_before = network_metrics[name_before]
                artist_metrics_before= metrics_before[metrics_before.artist_name == selected_artist]
                row['{0}_{1}_before_collaboration'.format(i,selected_metric)] = return_network_metric(artist_metrics_before)

            else:
                if error_switch:
                    row['{0}_{1}_before_collaboration'.format(i,selected_metric)]  = 'Error: No months before found'
                else:
                    row['{0}_{1}_before_collaboration'.format(i,selected_metric)]  = None


            # months after date
            name_after = 'network_metrics_after_{0}_{1}.csv'.format(tdate,split_unit)
            if name_after in network_metrics:
                metrics_after = network_metrics[name_after]
                artist_metrics_after = metrics_after[metrics_after.artist_name == selected_artist]
                row['{0}_{1}_after_collaboration'.format(i,selected_metric)]  = return_network_metric(artist_metrics_after)

            else:
                if error_switch:
                    row['{0}_{1}_after_collaboration'.format(i,selected_metric)] = 'Error: No months after found'
                else:
                    row['{0}_{1}_after_collaboration'.format(i,selected_metric)] = None

        return row

    return df.apply(add_query_metrics,axis=1)


def apply_add_additional_metrics(df,network_metrics,network_metrics_total,selected_metric,split_unit):

    def add_additional_metrics(row):
        tdate = row[split_unit]

        for i in ['artist_1','artist_2']:
            selected_artist = row[i]
            error_switch = False

            def return_network_metric(df):
                if len(df) == 0:
                    if error_switch:
                        return 'Artist not found in network'
                    else:
                        return None
                elif len(df) >1:
                    if error_switch:
                        return 'Error:'+str(len(df))+' artists found in network'
                    else:
                        return None
                else:
                    return df[selected_metric]

             # total
            artist_total_metrics = network_metrics_total[network_metrics_total.artist_name == selected_artist]
            row['{0}_{1}_uptil_today'.format(i,selected_metric)] = return_network_metric(artist_total_metrics)


            # months before date
            name_before = 'network_metrics_before_{0}_{1}.csv'.format(tdate,split_unit)
            if name_before in network_metrics:
                metrics_before = network_metrics[name_before]
                artist_metrics_before= metrics_before[metrics_before.artist_name == selected_artist]
                row['{0}_{1}_before_collaboration'.format(i,selected_metric)] = return_network_metric(artist_metrics_before)

            else:
                if error_switch:
                    row['{0}_{1}_before_collaboration'.format(i,selected_metric)]  = 'Error: No months before found'
                else:
                    row['{0}_{1}_before_collaboration'.format(i,selected_metric)]  = None


            # months after date
            name_after = 'network_metrics_after_{0}_{1}.csv'.format(tdate,split_unit)
            if name_after in network_metrics:
                metrics_after = network_metrics[name_after]
                artist_metrics_after = metrics_after[metrics_after.artist_name == selected_artist]
                row['{0}_{1}_after_collaboration'.format(i,selected_metric)]  = return_network_metric(artist_metrics_after)

            else:
                if error_switch:
                    row['{0}_{1}_after_collaboration'.format(i,selected_metric)] = 'Error: No months after found'
                else:
                    row['{0}_{1}_after_collaboration'.format(i,selected_metric)] = None

        return row

    return df.apply(add_additional_metrics,axis=1)

# --------------------------------------------------------------------------------

# Playlist Analysis

def remove_punctuation(name):
    import string
    return(''.join([i for i in name if i not in string.punctuation]).lower().strip())

def clean_artist_names(df,colname):
    import string
    df['cleaned_names'] = df[colname].apply(lambda x:''.join([i for i in x
                                                  if i not in string.punctuation])).str.lower().str.strip()
    return df
 # Convert to standard df
def convert_playlist_dict_to_frame(playlist_dict):
    playlist_dict = playlist_dict.to_frame().reset_index()
    playlist_dict.columns = ['stream_source_uri','COUNT']
    return playlist_dict

# Mapper function
def map_playlist_names(df,col_stream_source_uri,playlist_mapper):

    if col_stream_source_uri == 'index':
        df['playlist_id'] = df.index.astype(str).str[-22:]
        df['playlist_name'] = df.playlist_id.map(playlist_mapper.drop_duplicates(['id']).set_index('id')['name'])

        # return df.drop('playlist_id',axis=1)
        return df

    df['playlist_id'] = df[col_stream_source_uri].astype(str).str[-22:]
    df['playlist_name'] = df.playlist_id.map(playlist_mapper.drop_duplicates(['id']).set_index('id')['name'])

    # return df.drop('playlist_id',axis=1)
    return df

def get_years_on_key_playlists_dates(df,key_playlists,playlist_mapper):
        df = df.dropna(subset=['stream_source_uri'])
        df = df[df.stream_source_uri.str.len()>22]
        df['playlist_id'] = df.stream_source_uri.astype(str).str[-22:]

        df_filter = df[df.playlist_id.isin(key_playlists.playlist_id.tolist())]

        add_datetime_detail(df_filter)

        # Map names
        yearwise_key_playlists = df_filter.groupby(['stream_source_uri','year'])['date'].size().rename('COUNT')
        yearwise_key_playlists = map_playlist_names(yearwise_key_playlists.to_frame().reset_index(),'stream_source_uri',playlist_mapper)
        yearwise_key_playlists = yearwise_key_playlists[['playlist_name','playlist_id','year','COUNT']]

        key_playlists_2017 = yearwise_key_playlists[yearwise_key_playlists.year==2017]

        # Successful or not
        successful_artist = False
        if 2017 in df_filter.year.unique():

            successful_artist = True
            print('Successful playlists in 2017:\n',key_playlists_2017)

            # Only 2017 dates
            min_dates_2017 = df_filter[df_filter.year == 2017].groupby(['stream_source_uri'])['date'].min()


        else:
            min_dates_2017 = None

        # Success before 2017
        success_before_2017 = 0
        if successful_artist & (len(df_filter.year.unique())>1):
            warnings.warn( "This artist has been on successful playlists before 2017", stacklevel=2)
            print('Successful playlists before 2017:\n',yearwise_key_playlists[yearwise_key_playlists.year!=2017])
            success_before_2017 = 1


        output = {
        'success_before_2017':success_before_2017,
        'min_dates_2017':min_dates_2017,
        'successful_artist':successful_artist,
        'key_playlists_2017':key_playlists_2017
        }

        return output


def subset_before_success(df,key_playlists,playlist_mapper):

        gbfd = get_years_on_key_playlists_dates(df,key_playlists,playlist_mapper)

        success_before_2017 = gbfd['success_before_2017']
        min_dates_2017 = gbfd['min_dates_2017']
        successful_artist = gbfd['successful_artist']
        key_playlists_2017 = gbfd['key_playlists_2017']

        # For artists who were successful in 2017
        if successful_artist:

            if len(min_dates_2017)!=0:
                add_datetime_detail(df)
                ndf = df[df.date< min_dates_2017.to_frame().date.min()]

        # For unsuccessful artists - keep all available data including 2017
        else:
            ndf = df[:]


        output = {
        'success_before_2017':success_before_2017,
        'ndf':ndf,
        'successful_artist':successful_artist,
        'key_playlists_2017':key_playlists_2017
        }

        return output





# Generate entire feature set for an artist
def generate_feature_set(df, artist_name,key_playlists, new_artists,playlist_mapper,pca_dfs):
    warnings.resetwarnings()
    warnings.simplefilter('always')

    print(artist_name,' nrows:',len(df))

    # ----------------------------
    # Keep original artist_name for PCA
    artist_name_PCA = artist_name
    # ----------------------------

    artist_name = remove_punctuation(artist_name)


    # For country of Origin and Genre
#    new_artists = clean_artist_names(new_artists,'DISPLAY_NAME')

    # Make a copy for debugging
    df_original = df[:]

    # Add datetime detail
    add_datetime_detail(df_original)


    sbs = subset_before_success(df_original,key_playlists,playlist_mapper)
    df_original = sbs['ndf']
    success_before_2017 = sbs['success_before_2017']
    successful_artist = sbs['successful_artist']
    key_playlists_2017 = sbs['key_playlists_2017']

    print('nrows: subset_before_success:',len(df_original))
    print('successful artist:', successful_artist)



    # If no before data for successful artists
    if len(df_original) == 0:
        output_data = []
        # print('Successful Arist without before data')
        warnings.warn('Successful Artist without before data')


    else:

        # Get years of data
        data_years = df_original.year.unique()

        # Playlist
        # clean
        df_ssu_clean = df_original.dropna(subset = ['stream_source_uri'])
        stream_source_uri_rows_COUNT = len(df_ssu_clean)

        # --------------------------------------------------------------------
        #PCA

        pca_playlist_all = pca_dfs['pca_playlist_all']
        pca_playlist_uk = pca_dfs['pca_playlist_uk']
        pca_users = pca_dfs['pca_users']
        pca_locations = pca_dfs['pca_locations']

        # pca_playlists_all_artist_select = pca_playlist_all[pca_playlist_all.index == artist_name_PCA]
        # pca_playlists_uk_artist_select = pca_playlist_uk[pca_playlist_uk.index == artist_name_PCA]
        # pca_users_artists_select = pca_users[pca_users.index == artist_name_PCA]
        # pca_locations_artist_select = pca_locations[pca_locations.index == artist_name_PCA]

        # TEMPERORY FOR AWS MEMORY LIMIT  (LOADING PCA FROM SAVED CSVs)
        pca_playlists_all_artist_select = pca_playlist_all[pca_playlist_all.artist_name == artist_name_PCA]
        pca_playlists_uk_artist_select = pca_playlist_uk[pca_playlist_uk.artist_name == artist_name_PCA]
        pca_users_artists_select = pca_users[pca_users.artist_name == artist_name_PCA]
        pca_locations_artist_select = pca_locations[pca_locations.artist_name == artist_name_PCA]

        # Check for PCA data
        pca_playlists_all_binary = False
        pca_playlists_uk_binary = False
        pca_users_binary = False
        pca_locations_binary = False

        if len(pca_playlists_all_artist_select) > 0:
            pca_playlists_all_binary = True
            print('playlist data : all')
        if len(pca_playlists_uk_artist_select) > 0:
            pca_playlists_uk_binary = True
            print('playlist data : uk')
        if len(pca_users_artists_select) > 0:
            pca_users_binary = True
            print('playlist data : users')
        if len(pca_locations_artist_select) > 0:
            pca_locations_binary = True
            print('playlist data : locations')

        # --------------------------------------------------------------------

       # Appearance & Stream count on playlist - top 20 for each artist
        # ALL
        top20_all_playlists_raw = df_ssu_clean.stream_source_uri.value_counts().head(20)
        top20_all_playlists = convert_playlist_dict_to_frame(top20_all_playlists_raw)
        top20_all_playlists['PERCENTAGE'] = top20_all_playlists.COUNT/sum(top20_all_playlists.COUNT)

        #UK Only
        top20_spotify_uk_playlists = convert_playlist_dict_to_frame(df_ssu_clean[df_ssu_clean.stream_source_uri.str.contains('spotify:user:spotify_uk_:playlist')].stream_source_uri.value_counts().head(20))
        top20_spotify_uk_playlists['PERCENTAGE'] = top20_spotify_uk_playlists.COUNT/sum(top20_spotify_uk_playlists.COUNT)


        # Length of time on each playlist
        lot_all = convert_playlist_dict_to_frame(df_ssu_clean.groupby('stream_source_uri')['DateTime'].max()-df_ssu_clean.groupby('stream_source_uri')['DateTime'].min())
        top20_all_playlists['lot'] = top20_all_playlists.stream_source_uri.map(lot_all.set_index('stream_source_uri')['COUNT'])
        top20_spotify_uk_playlists['lot'] = top20_spotify_uk_playlists.stream_source_uri.map(lot_all.set_index('stream_source_uri')['COUNT'])


        # Country of origin
        coo = new_artists[new_artists.cleaned_names.str.contains(artist_name)].COUNTRY_OF_ORIGIN_CODE.unique().tolist()

        import math
        coo = [x for x in coo if str(x) != 'nan']

        # Genre
        genre = new_artists[new_artists.cleaned_names.str.contains(artist_name)].MAJOR_GENRE_CODE.unique().tolist()
        genre = [x for x in genre if str(x) != 'nan']
        # genre = np.array(genre)
        # genre = genre[genre!='nan']

        # Passion Score
        passion_score = round(len(df_original)/len(df_original.customer_id.unique()),2)

        # In case this is useful to show...
        # total streams
        total_streams = len(df_original)
        #total unique customers
        total_unique_users = len(df_original.customer_id.unique())

        ### User base
        # Gender percentage breakdown
        male_percentage = gender_percentage(df_original)

        # Age percentage breakdown
        age_percentage = age_percentages(df_original)

        # Most popular locations
        top20_locations = df_original.region_code.value_counts().head(20)

        ### Aggregate all metrics
        output_data = {
            'top20_all_pl_raw':top20_all_playlists_raw.to_dict(),
            'top20_all_playlists':  top20_all_playlists,
            'top20_spotify_uk_playlists':top20_spotify_uk_playlists,

            # 'pca_PL_all_artist_select':pca_playlists_all_artist_select,
            # 'pca_PL_uk_artist_select':pca_playlists_uk_artist_select,
            # 'pca_USERS_artists_select':pca_users_artists_select,
            # 'pca_locations_artist_select':pca_locations_artist_select,

            # pca binaries
            'pca_PL_all_binary':pca_playlists_all_binary,
            'pca_PL_uk_binary':pca_playlists_uk_binary,
            'pca_USERS_binary':pca_users_binary,
            'pca_locations_binary':pca_locations_binary,

            'coo':coo,
            'genre':genre,
            'passion_score':passion_score,
            'total_streams':total_streams,
            'total_unique_users':total_unique_users,
            'male_percentage':male_percentage,
            'age_percentage':age_percentage,
            'top20_locations': top20_locations,
            'success_before_2017':success_before_2017,
            'successful_artist':successful_artist,
            'data_years':data_years,
            'stream_source_uri_rows_COUNT':stream_source_uri_rows_COUNT,
            'key_PL_2017':key_playlists_2017.to_dict()

            }


        # Add playlist names to relevant dfs
        param = 'stream_source_uri'
        if len(output_data)>0:
            for i in output_data.keys():
                if 'playlist' in i:
                    output_data[i] = map_playlist_names(output_data[i],param,playlist_mapper)
            # [map_playlist_names(output_data[x],param,playlist_mapper) for x in output_data.keys() if 'playlist' in x]

    print('*' * 100)

    return output_data

# -----------------------------------------------------------------------------------------
# PCA ANALYSIS

def generate_pca(df,row_col,component_col,binary = False, pca_n_comp=10, orig_comp_head = 1000,uk_playlists_only = False,scale_data=True):


    # Split DF to deal with memory issues (break up df and process)

    def dropna_splitdf(df,split_size):
        split_df = np.array_split(df, split_size)
        for i in split_df:
            i = i.dropna(subset=[component_col])
        return pd.concat(split_df)


    split_df = True
    if split_df:
        df = dropna_splitdf(df,4)
    else:
        df = df.dropna(subset=[component_col])

    if uk_playlists_only:
        df = df.dropna(subset=['stream_source_uri'])
        df = df[df.stream_source_uri.str.contains('spotify:user:spotify_uk_:playlist')]

    subset = df[df[component_col].isin(df[component_col].value_counts().head(orig_comp_head).index)]
    pca_df = subset.groupby([row_col,component_col]).size().unstack(fill_value=0)
    playlist_component_names = pca_df.columns

    # Convert to binary dataset
    if binary ==True:
        pca_df[pca_df>0]= 1

    # Scale data before_PCA

    if scale_data & binary == False:
        from sklearn.preprocessing import scale
        pca_df = pd.DataFrame(scale(pca_df))

        print('Data Scaled!')


    pca_df_sparsity = np.count_nonzero(pca_df)/len(pca_df.values.flatten())

    X = pca_df
    from sklearn import decomposition

    pca = decomposition.PCA()
    pca.fit(X)
    decomposition.PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)
    pca.n_components = pca_n_comp
    X_reduced = pca.fit_transform(X)
    var_exp_cumsum = pca.explained_variance_ratio_.cumsum()
    var_exp_ratio = pca.explained_variance_ratio_
    var_exp = pca.explained_variance_
    mean = pca.mean_
    components =pca.components_

    pca_final = pd.DataFrame(X_reduced)
    pca_final.index = pca_df.index
    pca_final.columns = ['PCA_'+str(x+1)+'_{0}_uk_only_{1}'.format(component_col,str(uk_playlists_only)) for x in pca_final.columns]

    print('{0}_uk_only_{1}'.format(component_col,str(uk_playlists_only)),pca_final.shape)

    output = {'pca_df_sparisity':pca_df_sparsity,
              # 'pca_df':pca_df, # DF to be inputed to PCA
              'var_exp_cumsum':var_exp_cumsum,
              'pca_ouput_df':pca_final,
              'var_exp_ratio':var_exp_ratio,
              'var_exp':var_exp,
              'mean':mean,
              'components':components,
              'playlist_component_names':playlist_component_names
             }

    return output

def plot_pca_variance(df,col,item):
    pca_sorted = df.sort_values([col])
    x = range(len(pca_sorted['Unnamed: 0']))
    y = pca_sorted[col]

    fig = plt.figure(facecolor='white',figsize = (10,6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-')

    ax.set_title('{0}: {1} vs. N_COMPONENTS'.format(item.upper(),col.upper()))
    ax.set_xlabel('Components')
    ax.set_ylabel('{0}'.format(col.capitalize()))
    plt.show()


def plot_pca_variance_gradient(df,col,item):
    pca_sorted = df.sort_values([col])
    pca_sorted = pca_sorted.reset_index().sort_values(['Explained_Variance_CumSum'])
    pca_sorted['diff'] = pca_sorted[col].diff()
    pca_sorted['number_of_components'] = pca_sorted.index
    pca_sorted['gradient'] = pca_sorted['diff']/pca_sorted['number_of_components']
    plt.figure(figsize=(12,6))
    svd_plot = plt.plot(pca_sorted.gradient)
    plt.xlabel('Number of components')
    plt.ylabel('Gradient')
    plt.title("{0}: GRADIENTS OF EXPLAINED VARIANCE".format(item.upper()))
    plt.show()

    print('*'*100)

def get_component_weights(df,playlist_mapper,head_num,item):
    df = df.drop(['Unnamed: 0.1'],axis=1)
    df.index = df['Unnamed: 0']
    df = df.drop(['Unnamed: 0'],axis=1).transpose()

    component_weights = {}
    for i in df.columns:
        sort_vals = df.sort_values([i],ascending=False)[[i]].head(head_num)
        if item == 'playlists':
            sort_vals['id'] = sort_vals.index.astype(str).str[-22:]
            sort_vals['playlist_name'] = sort_vals['id'].map(playlist_mapper.set_index('id')['name'])
            sort_vals = sort_vals.drop(['id'],axis=1)
        component_weights['{0}_weights'.format(i)] = sort_vals

    return component_weights

def plot_weight_distribution(component_weights,how_many_components,item):
    fig, ax = plt.subplots()
#     df = component_weights['{0}_weights'.format(component_number)].reset_index()[[component_number]]
#     svd_plot = df.plot(ax=ax,figsize=(12,6))

    selected_component_weights = ['component_'+str(x)+'_weights' for x in range(1,how_many_components+1)]
    for i in selected_component_weights:
        df = component_weights['{0}'.format(i)].reset_index()[[i[:-8]]]
        df.plot(ax=ax,figsize=(12,6))

    ax.set_xlabel('Number of {0}'.format(item))
    ax.set_ylabel('Weights')
    plt.title('WEIGHTS OF TOP 100 {0} ({1} COMPONENTS)'.format(item.upper(),how_many_components))
    plt.show();
    return ax

def plot_cumsum_weights(component_weights,how_many_components,item):
    fig, ax = plt.subplots()
    selected_component_weights = ['component_'+str(x)+'_weights' for x in range(1,how_many_components+1)]
    for i in selected_component_weights:
        df = component_weights['{0}'.format(i)].reset_index()[[i[:-8]]].cumsum()
        df.plot(ax=ax,figsize=(12,6))

    ax.set_xlabel('Number of {0}'.format(item))
    ax.set_ylabel('Total Weight')
    plt.title('WEIGHTS OF TOP 100 {0} ({1} COMPONENTS)'.format(item.upper(),how_many_components))
    plt.show();

    return

def plot_gradient_weights(component_weights,how_many_components,item):

    fig, ax = plt.subplots()
    selected_component_weights = ['component_'+str(x)+'_weights' for x in range(1,how_many_components+1)]
    for i in selected_component_weights:
        component_number = i[:-8]
        recv = component_weights['{0}'.format(i)].reset_index()[[component_number]].cumsum()
        recv['number_of_items'] = recv.index
        recv['diff'] = recv[component_number].diff()
        recv['gradient'] = recv['diff']/recv['number_of_items']

        recv[['gradient']].plot(ax=ax,figsize=(12,6))

    ax.set_xlabel('Number of {0}'.format(item))
    ax.set_ylabel('Gradient')
    plt.title("GRADIENTS OF CUMULATIVE WEIGHT CURVE ({0} COMPONENTS)".format(how_many_components).upper())
    plt.show();

    return



# Aggregate above funcitons
def generate_pca_analysis(item,pca_all_variance,pca_component_weights,playlist_mapper,head_num,how_many_components):
    variance_plot = plot_pca_variance(pca_all_variance,'Explained_Variance_CumSum',item)
    gradient_variance = plot_pca_variance_gradient(pca_all_variance,'Explained_Variance_CumSum',item)
    component_weights = get_component_weights(pca_component_weights,playlist_mapper,head_num,item)
    # Specific Component
    weight_distribution = plot_weight_distribution(component_weights,how_many_components,item)
    cumsum_weights = plot_cumsum_weights(component_weights,how_many_components,item)
    gradient_weights = plot_gradient_weights(component_weights,how_many_components,item)

    return{'variance_plot':variance_plot,
           'gradient_variance':gradient_variance,
            'component_weights':component_weights,
            'weight_distribution':weight_distribution,
            'cumsum_weights':cumsum_weights,
            'gradient_weights':gradient_weights}


# ----------------------------------------------------------------------------
def important_component_user_demographics(important_components,pca_users_analysis,all_artists):
    selected_users = [pca_users_analysis['component_weights']['{0}_weights'.format(i)].index.unique().tolist() for i in important_components]
    selected_users_unique = pd.Series([item for sublist in selected_users for item in sublist]).unique()
    # print(len(selected_users_unique))
    age_breakdown_most_influental_users = age_percentages(all_artists[all_artists.customer_id.isin(selected_users)])
    gender_breakdown_most_influental_users = gender_percentage(all_artists[all_artists.customer_id.isin(selected_users)])

    return {'age_breakdown_most_influental_users':age_breakdown_most_influental_users,
           'gender_breakdown_most_influental_users':gender_breakdown_most_influental_users}


# -- ANALYZING TOP PCA COMPONENTS --
def pca_top_components(pca_analysis,how_many_components_to_keep,how_many_items_to_keep):
    all_component_weights = pca_analysis['component_weights']
    how_many_components_to_keep = how_many_components_to_keep
    top_components = {"component_{0}_weights".format(x):all_component_weights['component_{0}_weights'.format(x)] for x in range(1,how_many_components_to_keep+1)}

    if 'playlist_name' in top_components['component_1_weights'].columns:
        all_comps = [top_components[x].head(how_many_items_to_keep).playlist_name.unique().tolist() for x in top_components.keys()]

    else:
        all_comps = [top_components[x].head(how_many_items_to_keep).index.unique().tolist() for x in top_components.keys()]

    all_comps =[item for sublist in all_comps for item in sublist]
    all_comps_counts = pd.Series(all_comps).value_counts().rename('COUNT').to_frame().reset_index()


    return all_comps_counts

def get_age_gender_breakdown_of_top_USER_components(pca_top_components_df,df_with_all_data):
    filter_set = pca_top_components_df['index']
    age_breakdown = age_percentages(df_with_all_data[df_with_all_data['customer_id'].isin(filter_set)])
    gender_breakdown = gender_percentage(df_with_all_data[df_with_all_data['customer_id'].isin(filter_set)])

    return {'age_breakdown':age_breakdown,'gender_breakdown':gender_breakdown }



# -----------------------------------------------------------------------------------------------------------

def feature_set_all_artist(df,key_playlists, new_artists,playlist_mapper, pca_using_binary,pca_all_head_count,
    pca_uk_head_count, pca_user_head_count, pca_location_head_count, n_comp_pl_all,n_comp_pl_uk,ncomp_user,ncomp_location):

    # Generate PCA
    # Currently set to head(1000), n_comp = 10 for playlists, 50 for users
    pca_playlist_all = generate_pca(df,'artist_name','stream_source_uri', pca_using_binary,n_comp_pl_all,pca_all_head_count,False)
    pca_playlist_uk = generate_pca(df,'artist_name','stream_source_uri',pca_using_binary,n_comp_pl_uk,pca_uk_head_count,True)
    pca_users = generate_pca(df,'artist_name','customer_id', pca_using_binary,ncomp_user,pca_user_head_count,False)
    pca_locations = generate_pca(df,'artist_name','region_code',pca_using_binary,ncomp_location,pca_location_head_count,False)

    pca_dfs = {'pca_playlist_all':pca_playlist_all['pca_ouput_df'],
               'pca_playlist_uk':pca_playlist_uk['pca_ouput_df'],
               'pca_users':pca_users['pca_ouput_df'],
               'pca_locations':pca_locations['pca_ouput_df']
              }

    artists = df.artist_name.unique() # ALL ARTISTS
    output_all = {i:generate_feature_set(df[df.artist_name == i],i,
                    key_playlists, new_artists,playlist_mapper,pca_dfs) for i in artists}

    # Filter to remove artists with no data
    missing_data = [x for x in output_all.keys() if len(output_all[x]) == 0]
    filter_output_all = {x:output_all[x] for x in output_all.keys() if len(output_all[x])>0}

    # Put together final DF for regression - include only relevant variables

    # Add PCA at this stage, as all need to be calculated - use this method for when you need
    # to use all rows in df to do a calcuation (when subset for artist is only at the end) as
    # opposed to when a subset is required first, as processing times change
    final_dataframe =  (pd.DataFrame(filter_output_all).transpose()
    .merge(pd.DataFrame(pca_playlist_all['pca_ouput_df']),how='left',left_index=True,right_index=True)
    .merge(pd.DataFrame(pca_playlist_uk['pca_ouput_df']),how='left',left_index=True,right_index=True)
    .merge(pd.DataFrame(pca_users['pca_ouput_df']),how='left',left_index=True,right_index=True)
    .merge(pd.DataFrame(pca_locations['pca_ouput_df']),how='left',left_index=True,right_index=True)
    )

    # Correct names
    final_dataframe['adult'] = final_dataframe.age_percentage.apply(lambda x: x['Adult'])
    final_dataframe['youngadult'] = final_dataframe.age_percentage.apply(lambda x: x['YoungAdult'])
    final_dataframe['dependent'] = final_dataframe.age_percentage.apply(lambda x: x['Dependent'])
    final_dataframe['senior'] = final_dataframe.age_percentage.apply(lambda x: x['Senior'])


    final_output = {'missing_data':missing_data,
    'filter_output_all':filter_output_all,
    'final_dataframe':final_dataframe,
    'pca_dfs':pca_dfs

    }

    return final_output



def generate_correlations(df):
    c = df.corr().abs()
    s = c.unstack().sort_values()
    corr = sorted(s.items(),key = lambda x: x[1],  reverse=True)
    corr = [corr[x] for x in range(len(corr)) if corr[x][1]!=1]
    return{'s':s,'corr':corr}


def preprocess(df,selected_features = [],unbias = False,multiplier = 3,pca_binary=False,test_size = 0.4,normalize = False):
    final_df = df[:]
    final_df['coo'] = final_df.coo.map(str)
    final_df['genre'] = final_df.genre.map(str)
    #remove nas
    final_df = final_df[final_df.coo != '[ nan]']
    final_df = final_df[final_df.genre != '[ nan]']

    # Leave out artists who achieved success before 2017
    final_df = final_df[final_df.success_before_2017 == 0]

     # --------------------------------------------
    if pca_binary:
        final_df = final_df[(final_df.pca_locations_binary == True) &
                    (final_df.pca_PL_all_binary == True) &
                    # (final_df.pca_PL_uk_binary == True) &
                    (final_df.pca_USERS_binary == True)]
    # --------------------------------------------
    # Unbias
    if unbias == True:
        p1 = final_df[final_df.successful_artist == True]
        p2 = final_df[final_df.successful_artist == False].sample(round(len(p1)*multiplier))
        p3 = pd.concat([p1,p2])
        final_df = p3[:]
    # data frame charactertistics
    class_balance = final_df.successful_artist.value_counts()[0]/final_df.successful_artist.value_counts().sum()*100
    print('Class Balance = ',class_balance)
    print('nrow:',len(final_df.index))


    # Output to check quality of data
    check_na_df = final_df[:]




    # --------------------------------------------


    # features = [x for x in final_df.columns if x not in ['data_years','age_percentage','stream_source_uri_rows_COUNT']

    #             # NOTE: REDO CORRELATION MATRIX WITH FINAL FEATURE SET AND REMOVE COLLINEAR VARIABLES

    #             and 'binary' not in x
    #             and 'top20' not in x
    #             and 'total' not in x
    #             and 'uk_only_True' not in x
    # #             and 'stream' not in x
    #             and 'success' not in x
    #             and 'key_PL_2017' not in x
    #             # and 'PCA_2' not in x
    #             # and 'stream_source_uri_uk_only_False' in x
    #             # or 'p_' in x
    #            ]


    features = [

            # ---- ARTISTS ----
            #General
            'coo',
            'genre',
            'passion_score',

            # # #Network Centrality
            'artist_degree_centrality',
            'artist_betweeness_centrality',
            'artist_closeness_centrality',
            'artist_eigenvector_centrality',

            # # ---- USERS ----
            # # Listening Behaviour
            # # 'PCA_1_customer_id_uk_only_False',
            #  # 'PCA_2_customer_id_uk_only_False',
            # #  'PCA_3_customer_id_uk_only_False',
            # #  'PCA_4_customer_id_uk_only_False',

            # #  # Location
            'PCA_1_region_code_uk_only_False',
             'PCA_2_region_code_uk_only_False',
             'PCA_3_region_code_uk_only_False',
             'PCA_4_region_code_uk_only_False',
             'PCA_5_region_code_uk_only_False',

            #  # Gender
             'male_percentage',

            #  # Age
            'adult','youngadult','dependent','senior',




            # ---- PLAYLISTS ----

            'PCA_1_stream_source_uri_uk_only_False',
             'PCA_2_stream_source_uri_uk_only_False',
             'PCA_3_stream_source_uri_uk_only_False',
             'PCA_4_stream_source_uri_uk_only_False',
             'PCA_5_stream_source_uri_uk_only_False',
             'PCA_6_stream_source_uri_uk_only_False',
             'PCA_7_stream_source_uri_uk_only_False',
             'PCA_8_stream_source_uri_uk_only_False',
             'PCA_9_stream_source_uri_uk_only_False',
             'PCA_10_stream_source_uri_uk_only_False',

            # # Network Centrality
            # 'p_1_degree_centrality','p_1_eigenvector_centrality','p_1_closeness_centrality','p_1_betweeness_centrality',
            # 'p_2_degree_centrality','p_2_eigenvector_centrality','p_2_closeness_centrality','p_2_betweeness_centrality',
            # 'p_3_degree_centrality','p_3_eigenvector_centrality','p_3_closeness_centrality','p_3_betweeness_centrality',
            # 'p_4_degree_centrality','p_4_eigenvector_centrality','p_4_closeness_centrality','p_4_betweeness_centrality',
            # 'p_5_degree_centrality','p_5_eigenvector_centrality','p_5_closeness_centrality','p_5_betweeness_centrality',
            # 'p_6_degree_centrality','p_6_eigenvector_centrality','p_6_closeness_centrality','p_6_betweeness_centrality',
            # 'p_7_degree_centrality','p_7_eigenvector_centrality','p_7_closeness_centrality','p_7_betweeness_centrality',
            # 'p_8_degree_centrality','p_8_eigenvector_centrality','p_8_closeness_centrality','p_8_betweeness_centrality',
            # 'p_9_degree_centrality','p_1_eigenvector_centrality','p_9_closeness_centrality','p_9_betweeness_centrality',
            # 'p_10_degree_centrality','p_2_eigenvector_centrality','p_10_closeness_centrality','p_10_betweeness_centrality',
            # 'p_11_degree_centrality','p_3_eigenvector_centrality','p_11_closeness_centrality','p_11_betweeness_centrality',
            # 'p_12_degree_centrality','p_4_eigenvector_centrality','p_12_closeness_centrality','p_12_betweeness_centrality',
            # 'p_13_degree_centrality','p_5_eigenvector_centrality','p_13_closeness_centrality','p_13_betweeness_centrality',
            # 'p_14_degree_centrality','p_6_eigenvector_centrality','p_14_closeness_centrality','p_14_betweeness_centrality',
            # 'p_15_degree_centrality','p_7_eigenvector_centrality','p_15_closeness_centrality','p_15_betweeness_centrality',
            # 'p_16_degree_centrality','p_8_eigenvector_centrality','p_16_closeness_centrality','p_16_betweeness_centrality',
            # 'p_17_degree_centrality','p_4_eigenvector_centrality','p_17_closeness_centrality','p_17_betweeness_centrality',
            # 'p_18_degree_centrality','p_5_eigenvector_centrality','p_18_closeness_centrality','p_18_betweeness_centrality',
            # 'p_19_degree_centrality','p_6_eigenvector_centrality','p_19_closeness_centrality','p_19_betweeness_centrality',
            # 'p_20_degree_centrality','p_7_eigenvector_centrality','p_20_closeness_centrality','p_20_betweeness_centrality',




            # # Streams / Length of time on playlist
            #  'p_1_influence',
            #  'p_2_influence',
            #  'p_3_influence',
            #  'p_4_influence',
            #  'p_5_influence',
            #  'p_6_influence',
            #  'p_7_influence',
            #  'p_8_influence',
            #  'p_9_influence',
            #  'p_10_influence',
            #  'p_11_influence',
            #  'p_12_influence',
            #  'p_13_influence',
            #  'p_14_influence',
            #  'p_15_influence',
            #  'p_16_influence',
            #  'p_17_influence',
            #  'p_18_influence',
            #  'p_19_influence',
            #  'p_20_influence',

            #  'p_1_stream_count_lift',
            #  'p_2_stream_count_lift',
            #  'p_3_stream_count_lift',
            #  'p_4_stream_count_lift',
            #  'p_5_stream_count_lift',
            #  'p_6_stream_count_lift',
            #  'p_7_stream_count_lift',
            #  'p_8_stream_count_lift',
            #  'p_9_stream_count_lift',
            #  'p_10_stream_count_lift',
            #  'p_11_stream_count_lift',
            #  'p_12_stream_count_lift',
            #  'p_13_stream_count_lift',
            #  'p_14_stream_count_lift',
            #  'p_15_stream_count_lift',
            #  'p_16_stream_count_lift',
            #  'p_17_stream_count_lift',
            #  'p_18_stream_count_lift',
            #  'p_19_stream_count_lift',
            #  'p_20_stream_count_lift',

            #  'p_1_unique_user_lift',
            #  'p_2_unique_user_lift',
            #  'p_3_unique_user_lift',
            #  'p_4_unique_user_lift',
            #  'p_5_unique_user_lift',
            #  'p_6_unique_user_lift',
            #  'p_7_unique_user_lift',
            #  'p_8_unique_user_lift',
            #  'p_9_unique_user_lift',
            #  'p_10_unique_user_lift',
            #  'p_11_unique_user_lift',
            #  'p_12_unique_user_lift',
            #  'p_13_unique_user_lift',
            #  'p_14_unique_user_lift',
            #  'p_15_unique_user_lift',
            #  'p_16_unique_user_lift',
            #  'p_17_unique_user_lift',
            #  'p_18_unique_user_lift',
            #  'p_19_unique_user_lift',
            #  'p_20_unique_user_lift',

            #  'p_1_number_of_users',
            #  'p_2_number_of_users',
            #  'p_3_number_of_users',
            #  'p_4_number_of_users',
            #  'p_5_number_of_users',
            #  'p_6_number_of_users',
            #  'p_7_number_of_users',
            #  'p_8_number_of_users',
            #  'p_9_number_of_users',
            #  'p_10_number_of_users',
            #  'p_11_number_of_users',
            #  'p_12_number_of_users',
            #  'p_13_number_of_users',
            #  'p_14_number_of_users',
            #  'p_15_number_of_users',
            #  'p_16_number_of_users',
            #  'p_17_number_of_users',
            #  'p_18_number_of_users',
            #  'p_19_number_of_users',
            #  'p_20_number_of_users',



             'net_influence',
             'net_stream_count_lift',
             'net_unique_user_lift',
             'net_number_of_users',
             'net_degree_centrality',
             'net_betweeness_centrality',
             'net_closeness_centrality',
             'net_eigenvector_centrality',

             # 'recency_weighted_influence',
             # 'recency_weighted_stream_count_lift',
             # 'recency_weighted_unique_user_lift',
             # 'recency_weighted_number_of_users',
             # 'recency_weighted_avg_eigenvector_centrality',
             # 'recency_weighted_avg_closeness_centrality',
             # 'recency_weighted_avg_betweeness_centrality',
             # 'recency_weighted_avg_degree_centrality',

             # 'vc_weighted_influence',
             # 'vc_weighted_stream_count_lift',
             # 'vc_weighted_unique_user_lift',
             # 'vc_weighted_number_of_users',
             # 'vc_weighted_avg_eigenvector_centrality',
             # 'vc_weighted_avg_closeness_centrality',
             # 'vc_weighted_avg_betweeness_centrality',
             # 'vc_weighted_avg_degree_centrality'


             ]


    if len(selected_features)>0:
        features = selected_features

    print('*' * 100)

    # print('All features:',features)

    print('*' * 100)
    # --------------------------------------------
    # Impute Missing Data
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    for i in final_df.columns:
        if final_df[i].dtype == 'int64' or final_df[i].dtype == 'float64':
            final_df[i] = imp.fit_transform(final_df[[i]])

    #fill remaining nan with 0
    final_df = final_df.fillna(0)

    # --------------------------------------------
    # Scale data
    if normalize ==True:
        from sklearn.preprocessing import scale
        for i in final_df.columns:
            if final_df[i].dtype == 'int64' or final_df[i].dtype == 'float64':
                final_df[i] = scale(final_df[i])


    # --------------------------------------------

    # Check for highly correlated variables (>90%)
    c = final_df[features].corr().abs()
    s = c.unstack()
    corr = sorted(s.items(),key = lambda x: x[1],  reverse=True)
    corr = [corr[x] for x in range(len(corr)) if corr[x][1]!=1]
    # PCA 0 & PCA 2 found to be highly correlated - remove PCA 2
    # --------------------------------------------
    # Label Encoding for categorical variables
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()

    for i in final_df.columns:
        if i in ['coo','genre'] :
            final_df[i] = label_encoder.fit_transform(final_df[i].map(str))


    # --------------------------------------------
    from sklearn.model_selection import train_test_split
    X = final_df[features]
    y = final_df['successful_artist']
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=test_size, random_state=42)

    output =  {
    'X':X,'y':y,'X_train':X_train,'X_test':X_test, 'y_train':y_train, 'y_test':y_test, 'corr':corr,
    'final_df':final_df, 'features':features,'class_balance':class_balance}
    return output

def run_models(preprocess_output,try_all_classifiers= True, folds = 3):
    X = preprocess_output['X']
    y = preprocess_output['y']
    X_train = preprocess_output['X_train']
    X_test = preprocess_output['X_test']
    y_train = preprocess_output['y_train']
    y_test = preprocess_output['y_test']
    corr = preprocess_output['corr']
    features = preprocess_output['features']
    class_balance = preprocess_output['class_balance']

    print('Class Balance = ', class_balance)

    print('features:',features)
    print('*' * 100)

    accuracies_all = {}
    if try_all_classifiers == True:

          # XGBOOST
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score
        xgb_model = XGBClassifier(max_depth=10, n_estimators=3000, learning_rate=0.05)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        xgb_accuracy = accuracy_score(y_test, predictions)
        print("XGB Accuracy: %.2f%%" % (xgb_accuracy * 100.0))
        print('*' * 100)


        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn import linear_model
        from sklearn.model_selection import cross_val_score
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
        from sklearn.linear_model import LassoCV
        from sklearn.feature_selection import SelectFromModel

        classifiers = [
            linear_model.LogisticRegression(C=1e5),
            RandomForestClassifier( n_estimators=1000, max_features=4,oob_score=True),
            KNeighborsClassifier(3),
            # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
            # DecisionTreeClassifier(max_depth=5),
            # MLPClassifier(solver='sgd',activation = 'logistic', alpha=1, hidden_layer_sizes=(5, 2), random_state=1),
            # (alpha=1),
            #     # ,hidden_layer_sizes=[100], max_iter=2000,learning_rate_init = 1),
            # AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()
            ]


        def get_feature_names(model):
            mask = model.get_support() #list of booleans
            new_features = [] # The list of your K best features
            feature_names = X.columns.values
            for bool, feature in zip(mask, feature_names):
                if bool:
                    new_features.append(feature)

            return new_features



        all_model_features = {}


        accuracies_all = {}
        for clf in classifiers:
            clf.fit(X, y)
            # score = clf.score(X_test, y_test)
            cv_scores = cross_val_score(clf, X, y, cv=folds)
            accuracies_all[clf.__class__.__name__] = cv_scores.mean()
            print(clf.__class__.__name__)
            print('nrow:',len(X))
            # print(clf.__class__.__name__,'(test size = {0}):'.format(test_size),score)
            print("Cross Validation Accuracy ({0} folds) : %0.2f (+/- %0.2f)".format(folds) % (cv_scores.mean(), cv_scores.std() * 2))
            print('-' * 100)


    # print('*' * 100)
    # --------------------------------------------
    from sklearn import linear_model
    from sklearn.metrics import accuracy_score

    # --------------------------------------------

    # SELECTED MODEL: Random Forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import log_loss, precision_score,roc_curve

    np.random.seed(12)

    # GRIDSEARCH
    model = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
    from sklearn.model_selection import GridSearchCV, cross_val_score
    param_grid = {
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y_train)

    print('GRIDSEARCH')
    print('-'*10)
    print('Best Parameters: ',CV_rfc.best_params_)
    print('Best Score: ',CV_rfc.best_score_)

    # PRINT FINAL MODEL RESULT
    finalmodel = RandomForestClassifier(n_jobs=-1,max_features= CV_rfc.best_params_['max_features'] ,n_estimators=CV_rfc.best_params_['n_estimators'], oob_score = True)
    finalmodel.fit(X_train,y_train)

    from sklearn.metrics import confusion_matrix
    # VALIDATION SET SCORE
    y_pred = finalmodel.predict(X_test)
    train_score = finalmodel.score(X_train,y_train)
    validation_score = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    # print(confusion_mat)

    # # ROC curve
    from sklearn import metrics
    y_proba = finalmodel.predict_proba(X_test)
    y_score = [x[1] for x in y_proba]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
    roc_data = [fpr,tpr,thresholds]
    # print('y_proba:',y_proba)

    # LOG REG COMPARISON FOR BENCHMARK
    # LOG LOSS for logistic regression

    lr = linear_model.LogisticRegression(C=1e5)
    lr_model = lr.fit(X_train,y_train)
    # lr_coefs = lr.coef_
    lr_y_pred = lr.predict(X_test)
    lr_y_proba = lr.predict_proba(X_test)
    lr_y_score = [x[1] for x in lr_y_proba]
    lr_fpr, lr_tpr, lr_thresholds = metrics.roc_curve(y_test, lr_y_score,pos_label=1)
    lr_roc_data = [lr_fpr,lr_tpr,lr_thresholds]
    lr_coefs = {feature:coefficient for feature,coefficient in zip(features, lr.coef_[0])}

    print('Random Forest OOB Score: ',round(finalmodel.oob_score_,2)*100,"%")
    # print('Train Score: ',round(train_score,2)*100,"%")
    print('Validation Score: ',round(validation_score,2)*100,'%')

    # --------------------------------------------
    important_features = {feature:importance for feature,importance in zip(features, finalmodel.feature_importances_)}
    important_features = sorted(important_features.items(),key = lambda x: x[1],  reverse=True)

    # --------------------------------------------
    print('*' * 100)

    return {
    'features':features,
    'lr_roc_data':lr_roc_data,'lr_y_pred':lr_y_pred,'lr_y_score':lr_y_score,'lr_y_proba':lr_y_proba,'lr_coefs':lr_coefs,
    'roc_data':roc_data,'y_pred':y_pred,'y_score':y_score,'y_test':y_test,'y_proba':y_proba,
    'rf':finalmodel,'rf_important_features': important_features, 'confusion_matrix':confusion_mat,'accuracies_all':accuracies_all}
    # ,'classifier':classifier}


# --------------------------------------------------------------------


# PLAYLIST INFLUENCE
def playlist_influence_average(playlist_name,all_data):

    all_data = all_data.dropna(subset = ['stream_source_uri'])
    add_datetime_detail(all_data)

    selected_playlist = all_data[all_data.stream_source_uri == playlist_name]
    unique_artists = selected_playlist.sort_values(['DateTime']).drop_duplicates(['artist_name'])

    # Instead of row operation (to preserve the original + new data),
    # you could append each calculation to a list and sum the list
    def calculate_change(row):
        temp_date = row['DateTime']
        temp_artist = row['artist_name']

        temp_before = all_data[(all_data.DateTime < temp_date) & (all_data.artist_name == temp_artist)]
        temp_after = all_data[(all_data.DateTime > temp_date) & (all_data.artist_name == temp_artist)]

        row['change'] = len(temp_after) - len(temp_before)

        return row

    add_changes = unique_artists.apply(calculate_change,axis=1)
    avg_change = add_changes.change.mean().round()

    return avg_change

# Selected playlists only
def playlist_lift_list(df,key_playlists):
    df = df.dropna(subset = ['stream_source_uri'])
    df['playlist_id'] = df.stream_source_uri.astype(str).str[-22:]
    list_for_lift = df[df.playlist_id.isin(key_playlists.id)].stream_source_uri.unique()
    playlist_lift_key_playlists = {i:playlist_influence_average(i,df) for i in list_for_lift}
    return playlist_lift_key_playlists


def sort_dict(x,values = True):
    import operator
    sorted_x = sorted(x.items(), key=operator.itemgetter(1)) # sort by values
    if values == False:
        sorted_x = sorted(x.items(), key=operator.itemgetter(0)) # sort by keys
    return sorted_x
