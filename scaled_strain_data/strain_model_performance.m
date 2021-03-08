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

IDX = [1,2;3,4;5,6;7,9;10,11;12,15];


M_dnn_20 = readtable(append(result_directory,'model_scores_dnn_g20.txt'));
M_dnn_50 = readtable(append(result_directory,'model_scores_dnn_g50.txt'));
M_xgb_20 = readtable(append(result_directory,'model_scores_xgb_g20.txt'));
M_xgb_50 = readtable(append(result_directory,'model_scores_xgb_g50.txt'));

rock_TT_xgb_50 = table2array(readtable(append(result_directory,'rock_type_transfer_learning_score_matrix_xgb_g50.txt')));

scores = zeros(4,length(IDX));
A = zeros(4,length(IDX));
B = zeros(4,length(IDX));
C = zeros(4,length(IDX));
D = zeros(4,length(IDX));

for i=1:length(IDX)
    scores(1,i) = mean(table2array( M_xgb_50( IDX(i,1):IDX(i,2), 3 ) ));
    scores(2,i) = mean(table2array( M_dnn_50( IDX(i,1):IDX(i,2), 3 ) ));
    scores(3,i) = mean(table2array( M_xgb_20( IDX(i,1):IDX(i,2), 3 ) ));
    scores(4,i) = mean(table2array( M_dnn_20( IDX(i,1):IDX(i,2), 3 ) ));
    
end

Big_M = [table2array( M_xgb_50(:,3) ), table2array( M_dnn_50(:,3) ), table2array( M_dnn_20(:,3) ), table2array( M_dnn_20(:,3) )];
A = zeros(4,4,length(IDX));
%1
for i=1:4
    A(1,1,i) = Big_M(1,i);
    A(2,1,i) = Big_M(3,i);
    A(3,1,i) = Big_M(5,i);
    A(4,1,i) = Big_M(7,i);
    A(5,1,i) = Big_M(10,i);
    A(6,1,i) = Big_M(12,i);

    A(1,2,i) = Big_M(2,i);
    A(2,2,i) = Big_M(4,i);
    A(3,2,i) = Big_M(6,i);
    A(4,2,i) = Big_M(8,i);
    A(5,2,i) = Big_M(11,i);
    A(6,2,i) = Big_M(13,i);

    A(4,3,i) = Big_M(9,i);
    A(6,3,i) = Big_M(14,i);

    A(6,4,i) = Big_M(15,i);
    
end

T_dnn_20 = readtable(append(result_directory,'transfer_learning_score_matrix_dnn_g20.txt'));
T_xgb_20 = readtable(append(result_directory,'transfer_learning_score_matrix_xgb_g20.txt'));
T_xgb_50 = readtable(append(result_directory,'transfer_learning_score_matrix_xgb_g50.txt')); 


A(A==0)=nan;
figure()
set(figure(length(experiment_name)+1), 'Pos', [488, 342, 880, 480])

tiledlayout(1,2)
%subplot(1,2,1)
ax = nexttile;

short = {'Sandstone' 'Basalt' 'Monzonite' 'Granite' 'Shale' 'Limestone'};
x = [1:6];
z = [1.25,2.25,3.25,4.25,5.25,6.25];
plot(x, A(:,1,1), 'rs')%, 'MarkerFaceColor','red'); %plot(x, A(:,1,1), 'r^', 'MarkerFaceColor','red');
hold on
plot(z, A(:,1,2), 'bs')%, 'MarkerFaceColor','blue'); %plot(z, A(:,1,2), 'ro', 'MarkerFaceColor','red');
hold on
plot(x, A(:,2,1), 'ro')%, 'MarkerFaceColor','red'); %plot(x, A(:,2,1), 'g^', 'MarkerFaceColor','green');
hold on
plot(x, A(:,3,1), 'r*')%, 'MarkerFaceColor','red'); %plot(x, A(:,3,1), 'b^', 'MarkerFaceColor','blue');
hold on
plot(x, A(:,4,1), 'r^')%, 'MarkerFaceColor','red'); %plot(x, A(:,4,1), 'k^', 'MarkerFaceColor','black');
hold on
plot(z, A(:,2,2), 'bo')%, 'MarkerFaceColor','blue'); %plot(z, A(:,2,2), 'go', 'MarkerFaceColor','green');
hold on
plot(z, A(:,3,2), 'b*')%, 'MarkerFaceColor','blue'); %plot(z, A(:,3,2), 'bo', 'MarkerFaceColor','blue');
hold on
plot(z, A(:,4,2), 'b^')%, 'MarkerFaceColor','blue'); %plot(z, A(:,4,2), 'ko', 'MarkerFaceColor','black');
title('Test Scores on Low Resolution');
lgd = legend('xgb','dnn','Location','southwest'); % legend('xgb','dnn', 'Location', 'southwest');
ylabel('Model R^{2}');
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
xlim([0,6.75]);
ylim([0.1,1.0]);
hold on

%subplot(1,2,2)
ax = nexttile;

x = [1:6];
z = [1.25,2.25,3.25,4.25,5.25,6.25];
plot(x, A(:,1,3), 'rs')%, 'MarkerFaceColor','red'); %plot(x, A(:,1,3), 'r^', 'MarkerFaceColor','red');
hold on
plot(x, A(:,2,3), 'ro')%, 'MarkerFaceColor','red'); %plot(x, A(:,2,3), 'g^', 'MarkerFaceColor','green');
hold on
plot(x, A(:,3,3), 'r*')%, 'MarkerFaceColor','red'); %plot(x, A(:,3,3), 'b^', 'MarkerFaceColor','blue');
hold on
plot(x, A(:,4,3), 'r^')%, 'MarkerFaceColor','red'); %plot(x, A(:,4,3), 'k^', 'MarkerFaceColor','black');
hold on
plot(z, A(:,2,4), 'bo')%, 'MarkerFaceColor','blue'); %plot(z, A(:,2,4), 'go', 'MarkerFaceColor','green');
hold on
plot(z, A(:,3,4), 'b*')%, 'MarkerFaceColor','blue'); %plot(z, A(:,3,4), 'bo', 'MarkerFaceColor','blue');
hold on
plot(z, A(:,4,4), 'b^')%, 'MarkerFaceColor','blue'); %plot(z, A(:,4,4), 'ko', 'MarkerFaceColor','black');
hold on
plot(z, A(:,1,4), 'bs')%, 'MarkerFaceColor','blue'); %plot(z, A(:,1,4), 'ro', 'MarkerFaceColor','red');
title('Test Score on High Resolution');

% lgd = legend('xgb','dnn');
exp_names = ["Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"];
lg  = legend(exp_names,'Orientation','Horizontal','NumColumns',4); 
lg.Layout.Tile = 'North'; % <-- Legend placement with tiled layout

ylabel('Model R^{2}');
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
xlim([0,6.75]);
ylim([0.1,1.0]);
hold on
saveas(gcf, append(figure_directory,'Model_Test_Scores_',mname,'_g',string(rad),'0'), 'epsc')

% WE WANT TO HAVE MULTIPLE LEGENDS, USING TILES
% lgd2 = legend('Exp 1', 'Exp 2', 'Exp 3', 'Exp 4','Orientation','Horizontal');
% lgd2.Layout.Tile = "south";

%rock_type_transfer
figure()
subplot(3,2,1)
sandstone = rock_TT_xgb_50(1,:);
plot(x, sandstone, 'b-s');
ylabel('Model R^{2}');
title('Sandstone')
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
hold on

subplot(3,2,2)
basalt = [rock_TT_xgb_50(1,2),rock_TT_xgb_50(2,2:6)];
plot(x, basalt, 'b-s');
ylabel('Model R^{2}');
title('Basalt')
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
hold on

subplot(3,2,3)
monzonite = [reshape(rock_TT_xgb_50(1:3,3),[1,3]),rock_TT_xgb_50(3,4:6)];
plot(x, monzonite, 'b-s');
ylabel('Model R^{2}');
title('Monzonite')
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
hold on

subplot(3,2,4)
granite = [reshape(rock_TT_xgb_50(1:4,4),[1,4]),rock_TT_xgb_50(4,5:6)];
plot(x, granite, 'b-s');
ylabel('Model R^{2}');
title('Granite')
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
hold on

subplot(3,2,5)
shale = [reshape(rock_TT_xgb_50(1:5,5),[1,5]),rock_TT_xgb_50(5,6)];
plot(x, shale, 'b-s');
ylabel('Model R^{2}');
title('Shale')
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
hold on

subplot(3,2,6)
limestone = reshape(rock_TT_xgb_50(:,6),[1,6]);
plot(x, limestone, 'b-s');
ylabel('Model R^{2}');
title('Limestone')
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
hold on
saveas(gcf, append(figure_directory,'COMPILATION_Rock_Type_Transfer_Scores'), 'epsc')

figure()
x = [0.5:5.5];
p = plot(x, sandstone, 'b-s');
hold on
p = plot(x, basalt, 'r-s');
hold on
plot(x, monzonite, 'g-s');
hold on
p = plot(x, granite, 'y-s');
hold on
p = plot(x, shale, 'm-s');
hold on
p = plot(x, limestone, 'k-s');
hold on
xlim([0,6]);
ylim([0.2,0.9]);
ylabel('Model R^{2}');
title(append(cname,' R^{2} Scores for Transfer Learning with Rock Types'))
legend(short);
set(gca,'FontSize',16,'LineWidth',2,'Xtick',[0.5:5.5],'XTickLabel',short,'XTickLabelRotation',45)
set(gcf,'Position',[100 100 1400 700])
saveas(gcf, append(figure_directory,'Rock_Type_Transfer_Scores_',mname,'_g',string(rad),'0'), 'epsc')


%make a hist of strain localization

figure() %Not being saved

plot_list = ["WG04","MONZ05"];%,"WG04"]; %only compare 2 sets at a time using this method
%last = zeros(2,2);

for i=1: length(plot_list)
    nbins = 25;
    datastring = append('strains_curr_',plot_list(i),'_g',string(rad),'0.txt');
    D = readtable(append(data_directory,datastring));

    ep = table2array(D(:,2));
    
    counter = [1];
    for j = 1: length(ep)-1
        if ep(j) ~= ep(j+1)
            counter = [counter ; j];
        end
    end
    counter = [counter ; length(ep)];
    %last(i,1) = counter(end-1); %last(i,2) = counter(end);
    
    subplot(3,1,1)
    dn_p50_end = table2array(D((counter(end-1):counter(end)),10)); 
    histogram(dn_p50_end,nbins,'BinWidth',0.3, 'Normalization','probability');
    title('Strain Population of Contraction at Failure')
    ylabel('Probability')
    xlabel(' Median Contraction')
    hold on;
    legend(plot_list)
    
    subplot(3,1,2)
    dp_p50_end = table2array(D((counter(end-1):counter(end)),19));
    histogram(dp_p50_end,nbins,'BinWidth',0.7, 'Normalization','probability');
    title('Strain Population of Dilation at Failure')
    ylabel('Probability')
    xlabel('Median Dilation')
    hold on;
    legend(plot_list)
    
    subplot(3,1,3)
    cur_p50_end = table2array(D((counter(end-1):counter(end)),28));
    histogram(cur_p50_end,nbins,'BinWidth',0.2, 'Normalization','probability');
    title('Strain Population of Shear at Failure')
    ylabel('Probability')
    xlabel('Median Shear')
    hold on;
    legend(plot_list)
end
set(gcf,'Position',[100 100 1100 1100])

plot_lists = [["WG04","MONZ05"]; ["FBL02","GRS03"]];
for y = 1:length(plot_lists(1,:))
    loading_list = ["Early in Loading","at Failure"];
    for x = 1: length(loading_list)
        figure()

        plot_list = plot_lists(y,:);
        c_list = ["r","b"];
        for i=1: length(plot_list)
            nbins = 25;
            datastring = append('strains_curr_',plot_list(i),'_g',string(rad),'0.txt');
            D = readtable(append(data_directory,datastring));

            ep = table2array(D(:,2));

            counter = [1];
            for j = 1: length(ep)-1
                if ep(j) ~= ep(j+1)
                    counter = [counter ; j];
                end
            end
            counter = [counter ; length(ep)];
            %last(i,1) = counter(end-1); %last(i,2) = counter(end);

            if (loading_list(x) == "Early in Loading")
                dn_p50_end = table2array(D((counter(1):counter(2)),11));
                dp_p50_end = table2array(D((counter(1):counter(2)),20));
                cur_p50_end = table2array(D((counter(1):counter(2)),29));
            else
                dn_p50_end = table2array(D((counter(end-1):counter(end)),11));
                dp_p50_end = table2array(D((counter(end-1):counter(end)),20));
                cur_p50_end = table2array(D((counter(end-1):counter(end)),29));
            end

            subplot(3,1,1)
            mean_dn_end = mean(dn_p50_end);
            std_dev_dn_end = std(dn_p50_end);
            [h1,h1_edges] = histcounts(dn_p50_end,ceil(length(dn_p50_end)/20));%,'BinWidth',0.3, 'Normalization','probability');
            plot(h1_edges(1:end-1), h1, c_list(i), "Linewidth", 1.2);
            lg = legend(plot_list);
            hold on;
            errorbar(mean_dn_end,70,std_dev_dn_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_dn_end,70,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Contraction ',loading_list(x)));
            %xlim([-0.5,11.5])
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;

            subplot(3,1,2)
            mean_dp_end = mean(dp_p50_end);
            std_dev_dp_end = std(dp_p50_end);
            [h2,h2_edges] = histcounts(dp_p50_end,ceil(length(dp_p50_end)/20));%,'BinWidth',0.7, 'Normalization','probability');
            plot(h2_edges(1:end-1), h2, c_list(i), "Linewidth", 1.2);
            hold on;
            errorbar(mean_dp_end,30,std_dev_dp_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_dp_end,30,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Dilation ',loading_list(x)));
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;

            subplot(3,1,3)
            mean_cur_end = mean(cur_p50_end);
            std_dev_cur_end = std(cur_p50_end);
            [h3,h3_edges] = histcounts(cur_p50_end,ceil(length(cur_p50_end)/28));%,'BinWidth',0.2, 'Normalization','probability');
            length(h1)
            plot(h3_edges(1:end-1), h3, c_list(i), "Linewidth", 1.2);
            hold on;
            errorbar(mean_cur_end,40,std_dev_cur_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_cur_end,40,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Shear ',loading_list(x)));
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;
        end
    set(gcf,'Position',[100 100 1100 1100])
    %LEGEND
    end
end

loading_list = ["Early in Loading","at Failure"];
for x = 1: length(loading_list)
    figure()

    plot_list = ["WG04","MONZ05","FBL02","GRS03"];
    c_list = ["r","b", "r","b"];
    
    for i=1: length(plot_list)
        nbins = 25;
        datastring = append('strains_curr_',plot_list(i),'_g',string(rad),'0.txt');
        D = readtable(append(data_directory,datastring));

        ep = table2array(D(:,2));

        counter = [1];
        for j = 1: length(ep)-1
            if ep(j) ~= ep(j+1)
                counter = [counter ; j];
            end
        end
        counter = [counter ; length(ep)];
        %last(i,1) = counter(end-1); %last(i,2) = counter(end);

        if (loading_list(x) == "Early in Loading")
            dn_p50_end = table2array(D((counter(1):counter(2)),11));
            dp_p50_end = table2array(D((counter(1):counter(2)),20));
            cur_p50_end = table2array(D((counter(1):counter(2)),29));
        else
            dn_p50_end = table2array(D((counter(end-1):counter(end)),11));
            dp_p50_end = table2array(D((counter(end-1):counter(end)),20));
            cur_p50_end = table2array(D((counter(end-1):counter(end)),29));
        end
    end
    
    for i=1: length(plot_list)
        nbins = 25;
        datastring = append('strains_curr_',plot_list(i),'_g',string(rad),'0.txt');
        D = readtable(append(data_directory,datastring));

        ep = table2array(D(:,2));

        counter = [1];
        for j = 1: length(ep)-1
            if ep(j) ~= ep(j+1)
                counter = [counter ; j];
            end
        end
        counter = [counter ; length(ep)];
        %last(i,1) = counter(end-1); %last(i,2) = counter(end);

        if (loading_list(x) == "Early in Loading")
            dn_p50_end = table2array(D((counter(1):counter(2)),11));
            dp_p50_end = table2array(D((counter(1):counter(2)),20));
            cur_p50_end = table2array(D((counter(1):counter(2)),29));
        else
            dn_p50_end = table2array(D((counter(end-1):counter(end)),11));
            dp_p50_end = table2array(D((counter(end-1):counter(end)),20));
            cur_p50_end = table2array(D((counter(end-1):counter(end)),29));
        end

        if (i <= 2)
            subplot(3,2,1)
            mean_dn_end = mean(dn_p50_end);
            std_dev_dn_end = std(dn_p50_end);
            [h1,h1_edges] = histcounts(dn_p50_end,ceil(length(dn_p50_end)/20));%,'BinWidth',0.3, 'Normalization','probability');
            plot(h1_edges(1:end-1), h1, c_list(i), "Linewidth", 1.2);
            lg = legend(plot_list(1:2));
            hold on;
            errorbar(mean_dn_end,70,std_dev_dn_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_dn_end,70,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Contraction ',loading_list(x)));
            %xlim([-0.5,11.5])
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;

            subplot(3,2,3)
            mean_dp_end = mean(dp_p50_end);
            std_dev_dp_end = std(dp_p50_end);
            [h2,h2_edges] = histcounts(dp_p50_end,ceil(length(dp_p50_end)/20));%,'BinWidth',0.7, 'Normalization','probability');
            plot(h2_edges(1:end-1), h2, c_list(i), "Linewidth", 1.2);
            hold on;
            errorbar(mean_dp_end,30,std_dev_dp_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_dp_end,30,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Dilation ',loading_list(x)));
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;

            subplot(3,2,5)
            mean_cur_end = mean(cur_p50_end);
            std_dev_cur_end = std(cur_p50_end);
            [h3,h3_edges] = histcounts(cur_p50_end,ceil(length(cur_p50_end)/28));%,'BinWidth',0.2, 'Normalization','probability');
            length(h1)
            plot(h3_edges(1:end-1), h3, c_list(i), "Linewidth", 1.2);
            hold on;
            errorbar(mean_cur_end,40,std_dev_cur_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_cur_end,40,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Shear ',loading_list(x)));
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;
        else
            subplot(3,2,2)
            mean_dn_end = mean(dn_p50_end);
            std_dev_dn_end = std(dn_p50_end);
            [h1,h1_edges] = histcounts(dn_p50_end,ceil(length(dn_p50_end)/20));%,'BinWidth',0.3, 'Normalization','probability');
            plot(h1_edges(1:end-1), h1, c_list(i), "Linewidth", 1.2);
            lg = legend(plot_list(3:4));
            hold on;
            errorbar(mean_dn_end,70,std_dev_dn_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_dn_end,70,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Contraction ',loading_list(x)));
            %xlim([-0.5,11.5])
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;

            subplot(3,2,4)
            mean_dp_end = mean(dp_p50_end);
            std_dev_dp_end = std(dp_p50_end);
            [h2,h2_edges] = histcounts(dp_p50_end,ceil(length(dp_p50_end)/20));%,'BinWidth',0.7, 'Normalization','probability');
            plot(h2_edges(1:end-1), h2, c_list(i), "Linewidth", 1.2);
            hold on;
            errorbar(mean_dp_end,30,std_dev_dp_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_dp_end,30,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Dilation ',loading_list(x)));
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;

            subplot(3,2,6)
            mean_cur_end = mean(cur_p50_end);
            std_dev_cur_end = std(cur_p50_end);
            [h3,h3_edges] = histcounts(cur_p50_end,ceil(length(cur_p50_end)/28));%,'BinWidth',0.2, 'Normalization','probability');
            length(h1)
            plot(h3_edges(1:end-1), h3, c_list(i), "Linewidth", 1.2);
            hold on;
            errorbar(mean_cur_end,40,std_dev_cur_end,c_list(i),'horizontal', "Linewidth", 1.2, 'HandleVisibility', 'off');
            hold on;
            plot(mean_cur_end,40,append(c_list(i),'s'), "Linewidth", 1.2, 'HandleVisibility', 'off');
            title(append('Strain Population of Shear ',loading_list(x)));
            ylabel('Probability')
            xlabel('Mean Shear')
            hold on;
        end
    end
set(gcf,'Position',[100 100 1100 1100])
%LEGEND
end

