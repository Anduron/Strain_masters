close all
clearvars

%SWITCH FROM MEDIAN TO MEAN: INCOMPLETE

result_directory = 'strain_model_output_MS/';
data_directory = 'strain_data_MS/';
figure_directory = 'strain_project_figures/';
experiment_name = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"];
rad = 5;
mname = 'xgb';
cname = 'XGB ';
short = {'Sandstone' 'Basalt' 'Monzonite' 'Granite' 'Shale' 'Limestone'};

%new_c_order = {'r','g','b','c','m','y','k','#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F','#056927'};
c_ind = zeros([1,15]);
cmap = colormap('hsv');
for i = 1:15
    c_ind(i) = 17*i;
end
new_c_order = cmap(c_ind,:);

%treat the shap val data

shap_vals = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
shap_imps = []; %6x27, 27/3=9 values for each feature, 6 rocktypes

plot_list = ["FBL","ETNA","MONZ","WG","GRS","ANS"]; %[Sandstone,Basalt,Monzonite,Granite,Shale,Limestone]

for i = 1: length(plot_list)
    TF = contains(experiment_name,plot_list(i));
    temp = [];
    counter = 0;
    
    for i = 1: length(experiment_name)
        
        
        if TF(i) == 1
            counter = counter + 1;
            datastring = append('Shap_vals_xgb_',experiment_name(i),'_g',string(rad),'0.txt');
            
            SH = readtable(append(result_directory,datastring));
            
            temp = [temp; transpose(table2array(SH(:,2))) ];
            
        end
        
    end
    shap_vals = [shap_vals;mean(temp,1)];
    
    
end

shap_vals = normalize(shap_vals,2,'range');

figure()
x = [0.5:8.5];

stat = ["90^{th}","75^{th}","50^{th}","mean","25^{th}","10^{th}","\sigma","#","sum"];
tiledlayout(1,3); %tiledlayout(3,1) 

%for i = 1: length(plot_list)
%subplot(3,1,1)
ax = nexttile;
plot(x,shap_vals(:,1:9),'-s')
%plot(shap_vals(:,1:9),x,'-s')

xlim([0,9]);
%ylim([0,0.06]);
ylabel('Importance');
title('Contraction Features') %title('SHAP feature importance for contraction features')
%legend(short);
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[0.5:8.5],'XTickLabel',stat,'XTickLabelRotation',45)
    
hold on
%subplot(3,1,2)
ax = nexttile;
plot(x,shap_vals(:,10:18),'-s')
%plot(shap_vals(:,10:18),x,'-s')

xlim([0,9]);
%ylim([0,0.07]);
ylabel('Importance');
title('Dilation Features') %title('SHAP feature importance for dilation features')
%legend(short);
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[0.5:8.5],'XTickLabel',stat,'XTickLabelRotation',45)
    
hold on
%subplot(3,1,3)
ax = nexttile;
plot(x,shap_vals(:,19:27),'-s')
%plot(shap_vals(:,19:27),x,'-s')

xlim([0,9]);
%ylim([0,0.06]);
ylabel('Importance');
title('Shear Features') %title('SHAP feature importance for shear features')
%legend(short);
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[0.5:8.5],'XTickLabel',stat,'XTickLabelRotation',45)
    
hold on
sgt = sgtitle('SHAP Values for Strain Components');
sgt.FontSize = 18;

%end
set(gcf,'Position',[100 100 3000 1000])
lg  = legend(short,'Orientation','Horizontal','NumColumns',6); 
lg.Layout.Tile = 'North'; % <-- Legend placement with tiled layout

%ax = nexttile;
%ax = nexttile;
%ax = nexttile;

figfile = append(figure_directory,'shap_values_',mname,'_g',string(rad),'0_figure')
figfile2 = append(figure_directory,'shap_values_',mname,'_g',string(rad),'0_figure');

saveas(gcf, figfile, 'epsc')
saveas(gcf, figfile2, 'pdf')

%%%%

shap_vals = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
shap_imps = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
plot_list = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"]; %[Sandstone,Basalt,Monzonite,Granite,Shale,Limestone]

for i = 1: length(experiment_name)
    %TF = contains(experiment_name,plot_list(i));
    %temp = [];
    %counter = 0;
    
    datastring = append('Shap_vals_xgb_',experiment_name(i),'_g',string(rad),'0.txt');
            
    SH = readtable(append(result_directory,datastring));
    shap_vals = [shap_vals;transpose(table2array(SH(:,2)))];
    
end

shap_vals = normalize(shap_vals,2,'range');

figure()

x = [0.5:8.5];

stat = ["90^{th}","75^{th}","50^{th}","mean","25^{th}","10^{th}","\sigma","#","sum"];
tiledlayout(1,3); %tiledlayout(3,1) 

%for i = 1: length(plot_list)
%subplot(3,1,1)
ax = nexttile;
plot(x,shap_vals(:,1:9),'-s')
colormap(gca,'hsv');
%plot(shap_vals(:,1:9),x,'-s')

xlim([0,9]);
%ylim([0,0.06]);
ylabel('Importance');
title('Contraction Features') %title('SHAP feature importance for contraction features')
%legend(short);
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[0.5:8.5],'XTickLabel',stat,'XTickLabelRotation',45)
    
hold on
%subplot(3,1,2)
ax = nexttile;
plot(x,shap_vals(:,10:18),'-s')
colormap(gca,'hsv');
%plot(shap_vals(:,10:18),x,'-s')

xlim([0,9]);
%ylim([0,0.07]);
ylabel('Importance');
title('Dilation Features') %title('SHAP feature importance for dilation features')
%legend(short);
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[0.5:8.5],'XTickLabel',stat,'XTickLabelRotation',45)
    
hold on
%subplot(3,1,3)
ax = nexttile;
plot(x,shap_vals(:,19:27),'-s')
colormap(gca,'hsv');
%plot(shap_vals(:,19:27),x,'-s')

xlim([0,9]);
%ylim([0,0.06]);
ylabel('Importance');
title('Shear Features') %title('SHAP feature importance for shear features')
%legend(short);
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[0.5:8.5],'XTickLabel',stat,'XTickLabelRotation',45)
    
hold on
sgt = sgtitle('SHAP Values for Strain Components');
sgt.FontSize = 18;

%end
set(gcf,'Position',[100 100 3000 1000])
lg  = legend(plot_list,'Orientation','Horizontal','NumColumns',5); 
lg.Layout.Tile = 'North'; % <-- Legend placement with tiled layout

%ax = nexttile;
%ax = nexttile;
%ax = nexttile;

figfile = append(figure_directory,'shap_values_full_',mname,'_g',string(rad),'0_figure')
figfile2 = append(figure_directory,'shap_values_full_',mname,'_g',string(rad),'0_figure');

saveas(gcf, figfile, 'epsc')
saveas(gcf, figfile2, 'pdf')


figure()

shap_vals = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
shap_imps = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
plot_list = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"]; %[Sandstone,Basalt,Monzonite,Granite,Shale,Limestone]

for i = 1: length(plot_list)
    
    datastring = append('Shap_vals_xgb_',plot_list(i),'_g',string(rad),'0.txt');
            
    SH = readtable(append(result_directory,datastring));
    shap_vals = [shap_vals;transpose(table2array(SH(:,2)))];
    
end

shap_vals = normalize(shap_vals,2,'range');

x = [0.5:8.5];

tiledlayout(3,1); %tiledlayout(3,1) 

%for i = 1: length(plot_list)
%subplot(3,1,1)
ax = nexttile;

stat = categorical(["90^{th}","75^{th}","50^{th}","mean","25^{th}","10^{th}","\sigma","#","sum"]);
colororder(new_c_order);
shap_vals_statsum = shap_vals(:,1:9) + shap_vals(:,10:18) + shap_vals(:,19:27);
b = bar(stat,shap_vals_statsum,'stacked');
title('Cumulative Importance of Feature Statistics')
ylabel('Sum of Importances');
lg  = legend(plot_list,'Orientation','Horizontal','NumColumns',5); 
lg.Layout.Tile = 'North'; % <-- Legend placement with tiled layout

ax = nexttile;

component = categorical(["Contraction","Dilation","Shear"]);
colororder(new_c_order);
shap_vals_compsum = [];
shap_vals_compsum(:,1) = sum(shap_vals(:,1:9),2);
shap_vals_compsum(:,2) = sum(shap_vals(:,10:18),2);
shap_vals_compsum(:,3) = sum(shap_vals(:,19:27),2);
b = bar(component,shap_vals_compsum,'stacked');
title('Cumulative Importance of Strain Components')
ylabel('Sum of Importances');

ax = nexttile;

features = ["dn_p90","dn_p75","dn_p50","dn_mean","dn_p25","dn_p10","dn_std","dn_num","dn_sum","dp_p90","dp_p75","dp_p50","dp_mean","dp_p25","dp_p10","dp_std","dp_num","dp_sum","cur_p90","cur_p75","cur_p50","cur_mean","cur_p25","cur_p10","cur_std","cur_num","cur_sum"];

colororder(new_c_order);
[shap_vals_topn,index] = sort(sum(shap_vals),'descend');

%sort(shap_vals(:,index(1:9)), 'descend');

%top_nfeatures = features(index(1:9));
top_nfeatures = features;
top_nfeatures = strrep(top_nfeatures,"dn","Contraction");
top_nfeatures = strrep(top_nfeatures,"dp","Dilation");
top_nfeatures = strrep(top_nfeatures,"cur","Shear");
top_nfeatures = strrep(top_nfeatures,"_"," ");

top_nfeatures = strrep(top_nfeatures,"p90","90^{th}");
top_nfeatures = strrep(top_nfeatures,"p75","75^{th}");
top_nfeatures = strrep(top_nfeatures,"p50","50^{th}");
top_nfeatures = strrep(top_nfeatures,"p25","25^{th}");
top_nfeatures = strrep(top_nfeatures,"p10","10^{th}");
top_nfeatures = top_nfeatures(index(1:9));
top_nfeatures = reordercats(categorical(top_nfeatures),top_nfeatures);

b = bar(top_nfeatures,shap_vals(:,index(1:9)),'stacked');
title(append('Cumulative Importance of Top ',string(length(index(1:9))), ' Features' ))
ylabel('Sum of Importances');
%IMPORTANCE HERE IS VALUE OF NORMALIZED SHAPVALS
%ax = nexttile;

sgt = sgtitle('Cumulative SHAP Values');
sgt.FontSize = 18;

%end
set(gcf,'Position',[229    54.6    1022.4    724.8])

figfile = append(figure_directory,'shap_values_importance_stats_',mname,'_g',string(rad),'0_figure')
figfile2 = append(figure_directory,'shap_values_importance_stats_',mname,'_g',string(rad),'0_figure');

saveas(gcf, figfile, 'epsc')
saveas(gcf, figfile2, 'pdf')

%WEIGHTED SHAP VALS
figure()

shap_vals = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
shap_imps = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
plot_list = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"]; %[Sandstone,Basalt,Monzonite,Granite,Shale,Limestone]

for i = 1: length(plot_list)
    
    datastring = append('Shap_vals_xgb_',plot_list(i),'_g',string(rad),'0.txt');
    
    SH = readtable(append(result_directory,datastring));
    shap_vals = [shap_vals;transpose(table2array(SH(:,2)))];
    
end

weightstring = append('model_scores_xgb','_g',string(rad),'0.txt');
WH = readtable(append(result_directory,weightstring));
weights = table2array(WH(:,3));

shap_vals = normalize(shap_vals,2,'range');
shap_vals = shap_vals .* weights;


x = [0.5:8.5];

tiledlayout(3,1); %tiledlayout(3,1) 

%for i = 1: length(plot_list)
%subplot(3,1,1)
ax = nexttile;

stat = categorical(["90^{th}","75^{th}","50^{th}","mean","25^{th}","10^{th}","\sigma","#","sum"]);
colororder(new_c_order);
shap_vals_statsum = shap_vals(:,1:9) + shap_vals(:,10:18) + shap_vals(:,19:27);
b = bar(stat,shap_vals_statsum,'stacked');
title('Weighted Cumulative Importance of Feature Statistics')
ylabel('Sum of Importances');
lg  = legend(plot_list,'Orientation','Horizontal','NumColumns',5); 
lg.Layout.Tile = 'North'; % <-- Legend placement with tiled layout

ax = nexttile;

component = categorical(["Contraction","Dilation","Shear"]);
colororder(new_c_order);
shap_vals_compsum = [];
shap_vals_compsum(:,1) = sum(shap_vals(:,1:9),2);
shap_vals_compsum(:,2) = sum(shap_vals(:,10:18),2);
shap_vals_compsum(:,3) = sum(shap_vals(:,19:27),2);
b = bar(component,shap_vals_compsum,'stacked');
title('Weighted Cumulative Importance of Strain Components')
ylabel('Sum of Importances');

ax = nexttile;

features = ["dn_p90","dn_p75","dn_p50","dn_mean","dn_p25","dn_p10","dn_std","dn_num","dn_sum","dp_p90","dp_p75","dp_p50","dp_mean","dp_p25","dp_p10","dp_std","dp_num","dp_sum","cur_p90","cur_p75","cur_p50","cur_mean","cur_p25","cur_p10","cur_std","cur_num","cur_sum"];

colororder(new_c_order);
[shap_vals_topn,index] = sort(sum(shap_vals),'descend');

%sort(shap_vals(:,index(1:9)), 'descend');

%top_nfeatures = features(index(1:9));
top_nfeatures = features;
top_nfeatures = strrep(top_nfeatures,"dn","Contraction");
top_nfeatures = strrep(top_nfeatures,"dp","Dilation");
top_nfeatures = strrep(top_nfeatures,"cur","Shear");
top_nfeatures = strrep(top_nfeatures,"_"," ");

top_nfeatures = strrep(top_nfeatures,"p90","90^{th}");
top_nfeatures = strrep(top_nfeatures,"p75","75^{th}");
top_nfeatures = strrep(top_nfeatures,"p50","50^{th}");
top_nfeatures = strrep(top_nfeatures,"p25","25^{th}");
top_nfeatures = strrep(top_nfeatures,"p10","10^{th}");
top_nfeatures = top_nfeatures(index(1:9));
top_nfeatures = reordercats(categorical(top_nfeatures),top_nfeatures);

b = bar(top_nfeatures,shap_vals(:,index(1:9)),'stacked');
title(append('Weighted Cumulative Importance of Top ',string(length(index(1:9))), ' Features' ))
ylabel('Sum of Importances');
%IMPORTANCE HERE IS VALUE OF NORMALIZED SHAPVALS
%ax = nexttile;

sgt = sgtitle('Weighted Cumulative SHAP Values');
sgt.FontSize = 18;

%end
set(gcf,'Position',[7.4000   56.2000  724.0000  713.6000])

figfile = append(figure_directory,'shap_values_weighted_importance_stats_',mname,'_g',string(rad),'0_figure')
figfile2 = append(figure_directory,'shap_values_weighted_importance_stats_',mname,'_g',string(rad),'0_figure');

saveas(gcf, figfile, 'epsc')
saveas(gcf, figfile2, 'pdf')

%ROCK TYPE WITH WEIGHTED SHAP VALS

figure()


shap_vals = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
shap_imps = []; %6x27, 27/3=9 values for each feature, 6 rocktypes
plot_list = ["sandstone","basalt","monzonite","granite","shale","limestone"];

c_ind = zeros([1,length(plot_list)]);
cmap = colormap('hsv');
for i = 1:length(plot_list)
    c_ind(i) = floor(255/length(plot_list))*i;
end
new_c_order = cmap(c_ind,:);


for i = 1: length(plot_list)
    
    datastring = append('rock_type_transfer_shap_vals_xgb_',plot_list(i),'_g',string(rad),'0.txt');
            
    SH = readtable(append(result_directory,datastring));
    shap_vals = [shap_vals;transpose(table2array(SH(:,2)))];
    
end

weightstring = append('rock_type_transfer_learning_score_matrix_xgb_g',string(rad),'0.txt');
WH = readtable(append(result_directory,weightstring));

weights = diag(table2array(WH));

shap_vals = normalize(shap_vals,2,'range');
shap_vals = shap_vals .* weights;
x = [0.5:8.5];

tiledlayout(3,1); %tiledlayout(3,1) 

%for i = 1: length(plot_list)
%subplot(3,1,1)
ax = nexttile;

stat = categorical(["90^{th}","75^{th}","50^{th}","mean","25^{th}","10^{th}","\sigma","#","sum"]);
colororder(new_c_order);
shap_vals_statsum = shap_vals(:,1:9) + shap_vals(:,10:18) + shap_vals(:,19:27);
b = bar(stat,shap_vals_statsum,'stacked');
title('Weighted Cumulative Importance of Feature Statistics')
ylabel('Sum of Importances');
lg  = legend(plot_list,'Orientation','Horizontal','NumColumns',6); 
lg.Layout.Tile = 'North'; % <-- Legend placement with tiled layout

ax = nexttile;

component = categorical(["Contraction","Dilation","Shear"]);
colororder(new_c_order);
shap_vals_compsum = [];
shap_vals_compsum(:,1) = sum(shap_vals(:,1:9),2);
shap_vals_compsum(:,2) = sum(shap_vals(:,10:18),2);
shap_vals_compsum(:,3) = sum(shap_vals(:,19:27),2);
b = bar(component,shap_vals_compsum,'stacked');
title('Weighted Cumulative Importance of Strain Components')
ylabel('Sum of Importances');

ax = nexttile;

features = ["dn_p90","dn_p75","dn_p50","dn_mean","dn_p25","dn_p10","dn_std","dn_num","dn_sum","dp_p90","dp_p75","dp_p50","dp_mean","dp_p25","dp_p10","dp_std","dp_num","dp_sum","cur_p90","cur_p75","cur_p50","cur_mean","cur_p25","cur_p10","cur_std","cur_num","cur_sum"];

colororder(new_c_order);
[shap_vals_topn,index] = sort(sum(shap_vals),'descend');

%sort(shap_vals(:,index(1:9)), 'descend');

%top_nfeatures = features(index(1:9));
top_nfeatures = features;
top_nfeatures = strrep(top_nfeatures,"dn","Contraction");
top_nfeatures = strrep(top_nfeatures,"dp","Dilation");
top_nfeatures = strrep(top_nfeatures,"cur","Shear");
top_nfeatures = strrep(top_nfeatures,"_"," ");

top_nfeatures = strrep(top_nfeatures,"p90","90^{th}");
top_nfeatures = strrep(top_nfeatures,"p75","75^{th}");
top_nfeatures = strrep(top_nfeatures,"p50","50^{th}");
top_nfeatures = strrep(top_nfeatures,"p25","25^{th}");
top_nfeatures = strrep(top_nfeatures,"p10","10^{th}");
top_nfeatures = top_nfeatures(index(1:9));
top_nfeatures = reordercats(categorical(top_nfeatures),top_nfeatures);

b = bar(top_nfeatures,shap_vals(:,index(1:9)),'stacked');
title(append('Weighted Cumulative Importance of Top ',string(length(index(1:9))), ' Features' ))
ylabel('Sum of Importances');
%IMPORTANCE HERE IS VALUE OF NORMALIZED SHAPVALS
%ax = nexttile;

sgt = sgtitle('Weighted Cumulative SHAP Values');
sgt.FontSize = 18;

%end
set(gcf,'Position',[763.4000   56.2000  727.2000  713.6000])

figfile = append(figure_directory,'shap_values_weighted_rock_type_importance_stats_',mname,'_g',string(rad),'0_figure')
figfile2 = append(figure_directory,'shap_values_weighted_rock_type_importance_stats_',mname,'_g',string(rad),'0_figure');

saveas(gcf, figfile, 'epsc')
saveas(gcf, figfile2, 'pdf')
