close all
clearvars

%SWITCH FROM MEDIAN TO MEAN: COMPLETE

result_directory = 'strain_model_output_MS/';
data_directory = 'strain_data_MS/';
figure_directory = 'strain_project_figures/';
experiment_name = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"];
rad = 5;
mname = 'xgb';
cname = 'XGBoost ';

IDX = [1,2;3,4;5,6;7,9;10,11;12,15];

M_dnn_20 = readtable(append(result_directory,'model_scores_dnn_g20.txt'));
M_dnn_50 = readtable(append(result_directory,'model_scores_dnn_g50.txt'));
M_xgb_20 = readtable(append(result_directory,'model_scores_xgb_g20.txt'));
M_xgb_50 = readtable(append(result_directory,'model_scores_xgb_g50.txt'));


figure()

plot_list = ["FBL02", "ETNA01", "MONZ05", "WG01", "GRS03", "ANS04"]; %experiment_name = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"];
for i = 1: length(plot_list)


    subplot(3,2,i)
    resultstring = append('TEST_prediction_xgb_',plot_list(i),'_g',string(rad),'0.txt');
    %datastring = append('TEST_prediction_xgb_',plot_list(i),'_g',string(rad),'0.txt');
    R = load(append(result_directory,resultstring));
    %D = readtable(append(data_directory,datastring));
    ep = R(:,2);
    pred = R(:,3);
    x = linspace(1,length(ep),length(ep));

    counter = [1];
    for j = 1: length(ep)-1
        if ep(j) ~= ep(j+1)
            counter = [counter ; j];
        end
    end
    counter = [counter ; length(ep)];

    ep_list = [];
    pred_means = [];
    pred_stdev = [];

    for k = 1: length(counter)-1
        ep_list = [ep_list;counter(k)+round((counter(k+1) - counter(k))/2)];
        pred_means = [pred_means ; mean( pred( counter(k):counter(k+1) ) )];
        pred_stdev = [pred_stdev ; std( pred( counter(k):counter(k+1) ) )];
    end


    p = plot(x,pred, x,ep);
    set(p,{'LineWidth'},{0.7;1.3})
    p(1).Color = [0.05 0.45 0.98 0.5]; %[0.13 0.43 0.90]; [0.39 0.88 0.15]
    p(2).Color = [0.96 0.5 0.1]; %[0.91 0.41 0.17];
    hold on
    errorbar(ep_list,pred_means,pred_stdev,'-s','MarkerSize',5,'MarkerFaceColor',[1,0,0],'Color',[1,0,0],'LineWidth',1.3); %[0.8 0.2 0.7]
    title(append(plot_list(i),', R^2 = ',string(round(table2array(M_xgb_50( find( contains(experiment_name,plot_list(i)) ),3)) ,2)) ));

    if i == 1    
        legend('Test Prediction','Observed Strain','Mean Model Output','Location','northwest');
    end

    ylabel('Normalized Axial Strain');
    xlabel('Experiment Sample');
    axis([0 length(ep)+50 -0.05 1.05])
    hold on
    hold off
end
sgt = sgtitle(append(cname,' Test Data Predicted Vs Observed Axial Strain'));
sgt.FontSize = 18;

set(gcf,'Position',[100 100 1200 2000])
figfile = append(figure_directory,'IMAGE_COMPILATION_model_test_prediction_',mname,'_g',string(rad),'0_figure')
%figfile2 = append('figfiles/','model_prediction_',mname,'_',plot_list(i),'_g',string(rad),'_figure');


saveas(gcf, figfile, 'epsc')
%saveas(gcf, figfile2, 'fig')

figure()

plot_list = ["GRS03","ETNA01","MONZ05"]; %["ANS02","ANS03","FBL02"]; %["WG04","GRS02","ANS04"]; %["WG01","WG02","MONZ04"];%["FBL01","FBL02","ETNA02"];%["ANS02","ANS03","ANS05"]; %experiment_name = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"];
for i = 1: length(plot_list)
    resultstring = append('result_',mname,'_',plot_list(i),'_g',string(rad),'0.txt');
    resultstring2 = append('TEST_prediction_xgb_',plot_list(i),'_g',string(rad),'0.txt');
    datastring = append('strains_curr_',plot_list(i),'_g',string(rad),'0.txt');
    R = load(append(result_directory,resultstring));
    R2 = load(append(result_directory,resultstring2));
    D = readtable(append(data_directory,datastring));

    ep = R(:,2);
    pred = R(:,1);
    ep2 = R2(:,2);
    pred2 = R2(:,3);

    dn_mean = table2array(D(:,11)); %dn_p50 = table2array(D(:,10));
    dp_mean = table2array(D(:,20)); %dp_p50 = table2array(D(:,19));
    cur_mean = table2array(D(:,29)); %cur_p50 = table2array(D(:,28));
    dn_mean = dn_mean(~isnan(dn_mean)); %dn_p50 = dn_p50(~isnan(dn_p50));
    dp_mean = dp_mean(~isnan(dp_mean)); %dp_p50 = dp_p50(~isnan(dp_p50));
    cur_mean = cur_mean(~isnan(cur_mean)); %cur_p50 = cur_p50(~isnan(cur_p50));

    x = linspace(1,length(ep),length(ep));

    counter = [1];
    for j = 1: length(ep)-1
        if ep(j) ~= ep(j+1)
            counter = [counter ; j];
        end
    end
    counter = [counter ; length(ep)];

    eps = [];
    ep_list = [];
    pred_means = [];
    pred_stdev = [];
    input_dn_mean_means = []; %input_dn_p50_means = [];
    input_dp_mean_means = []; %input_dp_p50_means = [];
    input_cur_mean_means = []; %input_cur_p50_means = [];

    for k = 1: length(counter)-1
        eps = [eps; ep(counter(k+1))];
        ep_list = [ep_list;counter(k)+round((counter(k+1) - counter(k))/2)];
        pred_means = [pred_means ; mean( pred( counter(k):counter(k+1) ) )];
        pred_stdev = [pred_stdev ; std( pred( counter(k):counter(k+1) ) )];
        input_dn_mean_means = [input_dn_mean_means ; mean( dn_mean( counter(k):counter(k+1) ) )];
        input_dp_mean_means = [input_dp_mean_means ; mean( dp_mean( counter(k):counter(k+1) ) )];
        input_cur_mean_means = [input_cur_mean_means ; mean( cur_mean( counter(k):counter(k+1) ) )];
        %input_dn_p50_means = [input_dn_p50_means ; mean( dn_p50( counter(k):counter(k+1) ) )];
        %input_dp_p50_means = [input_dp_p50_means ; mean( dp_p50( counter(k):counter(k+1) ) )];
        %input_cur_p50_means = [input_cur_p50_means ; mean( cur_p50( counter(k):counter(k+1) ) )];
    end
    %eps = [eps;ep(counter(k+1))]

    subplot(3,2,i+(i-1))

    plot( eps,input_dn_mean_means, '-rs' ); %plot( eps,input_dn_p50_means, '-rs' );
    hold on
    plot( eps,input_dp_mean_means, '-bs' ); %plot( eps,input_dp_p50_means, '-bs' );
    hold on
    plot( eps,input_cur_mean_means, '-gs' ); %plot( eps,input_cur_p50_means, '-gs' );
    hold on

    ax_vals = [];
    ax_vals = [ax_vals ; max(input_dn_mean_means)]; %ax_vals = [ax_vals ; max(input_dn_p50_means)];
    ax_vals = [ax_vals ; max(input_dp_mean_means)]; %ax_vals = [ax_vals ; max(input_dp_p50_means)];
    ax_vals = [ax_vals ; max(input_cur_mean_means)]; %ax_vals = [ax_vals ; max(input_cur_p50_means)];

    title(append('Strain Evolution of Feature Mean in: ',plot_list(i) ));
    if i == 1
        legend('Mean Contraction','Mean Dilation','Mean Shear','Location','northwest') %legend('dn\_p50','dp\_p50','cur\_p50','Location','northwest');
    end
    ylabel('Mean Strain Magnitude');
    xlabel('Normalized Axial Strain');
    axis([-0.01 eps(end)*1.02 0 1.05*max(ax_vals)])
    %axis()


    x2 = linspace(1,length(ep2),length(ep2));

    counter = [1];
    for j = 1: length(ep2)-1
        if ep2(j) ~= ep2(j+1)
            counter = [counter ; j];
        end
    end
    counter = [counter ; length(ep2)];

    eps2 = [];
    ep_list2 = [];
    pred_means2 = [];
    pred_stdev2 = [];

    for k = 1: length(counter)-1
        eps2 = [eps2; ep2(counter(k+1))];
        ep_list2 = [ep_list2;counter(k)+round((counter(k+1) - counter(k))/2)];
        pred_means2 = [pred_means2 ; mean( pred2( counter(k):counter(k+1) ) )];
        pred_stdev2 = [pred_stdev2 ; std( pred2( counter(k):counter(k+1) ) )];
    end

    subplot(3,2,2*i)
    p = plot(x2,pred2, x2,ep2);
    set(p,{'LineWidth'},{0.7;1.3})
    p(1).Color = [0.05 0.45 0.98 0.5]; %[0.13 0.43 0.90]; [0.39 0.88 0.15]
    p(2).Color = [0.96 0.5 0.1]; %[0.91 0.41 0.17];
    hold on
    errorbar(ep_list2,pred_means2,pred_stdev2,'-s','MarkerSize',5,'MarkerFaceColor',[1,0,0],'Color',[1,0,0],'LineWidth',1.3); %[0.8 0.2 0.7]
    title(append(cname,'Predicted Vs Observed Axial Strain ',plot_list(i),', R^2 = ',string(round(table2array(M_xgb_50( find( contains(experiment_name,plot_list(i)) ),3)) ,2)) ));
    if i == 1
    legend('Model Prediction','Observed Strain','Mean Model Output','Location','northwest');
    end 
    ylabel('Normalized Axial Strain');
    xlabel('Experiment Sample');
    axis([0 length(ep2)+50 -0.05 1.05]);
    hold on
    hold off
end

set(gcf,'Position',[100 100 1200 2000])
%set(gcf,'Position',[100 100 1000 2000])
figfile = append(figure_directory,'input_data_and_model_prediction_',mname,'_g',string(rad),'0_figure')
figfile2 = append(figure_directory,'input_data_and_model_prediction_',mname,'_g',string(rad),'0_figure');


saveas(gcf, figfile, 'epsc')
saveas(gcf, figfile2, 'pdf')

figure()

plot_list = ["FBL02"]; %experiment_name = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"];
rads = [5,2];
resolution = ["Low","High"];
for i = 1: length(rads)
    resultstring = append('result_',mname,'_',plot_list,'_g',string(rads(i)),'0.txt');
    resultstring2 = append('TEST_prediction_xgb_',plot_list,'_g',string(rads(i)),'0.txt');
    datastring = append('strains_curr_',plot_list,'_g',string(rads(i)),'0.txt');
    R = load(append(result_directory,resultstring));
    R2 = load(append(result_directory,resultstring2));
    D = readtable(append(data_directory,datastring));

    ep = R(:,2);
    pred = R(:,1);
    ep2 = R2(:,2);
    pred2 = R2(:,3);

    dn_mean = table2array(D(:,11)); %dn_p50 = table2array(D(:,10));
    dp_mean = table2array(D(:,20)); %dp_p50 = table2array(D(:,19));
    cur_mean = table2array(D(:,29)); %cur_p50 = table2array(D(:,28));
    dn_mean = dn_mean(~isnan(dn_mean)); %dn_p50 = dn_p50(~isnan(dn_p50));
    dp_mean = dp_mean(~isnan(dp_mean)); %dp_p50 = dp_p50(~isnan(dp_p50));
    cur_mean = cur_mean(~isnan(cur_mean)); %cur_p50 = cur_p50(~isnan(cur_p50));

    x = linspace(1,length(ep),length(ep));

    counter = [1];
    for j = 1: length(ep)-1
        if ep(j) ~= ep(j+1)
            counter = [counter ; j];
        end
    end
    counter = [counter ; length(ep)];

    eps = [];
    ep_list = [];
    pred_means = [];
    pred_stdev = [];
    input_dn_mean_means = []; %input_dn_p50_means = [];
    input_dp_mean_means = []; %input_dp_p50_means = [];
    input_cur_mean_means = []; %input_cur_p50_means = [];
    input_dn_mean_stdev = []; %input_dn_p50_stdev = [];
    input_dp_mean_stdev = []; %input_dp_p50_stdev = [];
    input_cur_mean_stdev = []; %input_cur_p50_stdev = [];

    for k = 1: length(counter)-1
        eps = [eps; ep(counter(k+1))];
        ep_list = [ep_list;counter(k)+round((counter(k+1) - counter(k))/2)];
        pred_means = [pred_means ; mean( pred( counter(k):counter(k+1) ) )];
        pred_stdev = [pred_stdev ; std( pred( counter(k):counter(k+1) ) )];
        input_dn_mean_means = [input_dn_mean_means ; mean( dn_mean( counter(k):counter(k+1) ) )]; %input_dn_p50_means = [input_dn_p50_means ; mean( dn_p50( counter(k):counter(k+1) ) )];
        input_dp_mean_means = [input_dp_mean_means ; mean( dp_mean( counter(k):counter(k+1) ) )]; %input_dp_p50_means = [input_dp_p50_means ; mean( dp_p50( counter(k):counter(k+1) ) )];
        input_cur_mean_means = [input_cur_mean_means ; mean( cur_mean( counter(k):counter(k+1) ) )]; %input_cur_p50_means = [input_cur_p50_means ; mean( cur_p50( counter(k):counter(k+1) ) )];
        input_dn_mean_stdev = [input_dn_mean_stdev ; std( dn_mean( counter(k):counter(k+1) ) )]; %input_dn_p50_stdev = [input_dn_p50_stdev ; std( dn_p50( counter(k):counter(k+1) ) )];
        input_dp_mean_stdev = [input_dp_mean_stdev ; std( dp_mean( counter(k):counter(k+1) ) )]; %input_dp_p50_stdev = [input_dp_p50_stdev ; std( dp_p50( counter(k):counter(k+1) ) )];
        input_cur_mean_stdev = [input_cur_mean_stdev ; std( cur_mean( counter(k):counter(k+1) ) )]; %input_cur_p50_stdev = [input_cur_p50_stdev ; std( cur_p50( counter(k):counter(k+1) ) )];
    end
    %eps = [eps;ep(counter(k+1))]

    subplot(2,2,i+(i-1))

    %errorbar(ep_list2,pred_means2,pred_stdev2,'-s','MarkerSize',5,'MarkerFaceColor',[1,0,0],'Color',[1,0,0],'LineWidth',1.3); %[0.8 0.2 0.7]
    %plot( eps,input_dn_p50_means, '-rs' );
    %hold on
    errorbar(eps, input_dn_mean_means,input_dn_mean_stdev, '-rs')%errorbar(eps, input_dn_p50_means,input_dn_p50_stdev, '-rs')%,'MarkerSize',5,'MarkerFaceColor','Red','LineWidth',1.3);
    hold on
    %plot( eps,input_dp_p50_means, '-bs' );
    %hold on
    errorbar( eps, input_dp_mean_means,input_dp_mean_stdev, '-bs')%errorbar( eps, input_dp_p50_means,input_dp_p50_stdev, '-bs')%,'MarkerSize',5,'MarkerFaceColor','Blue','LineWidth',1.3);
    hold on
    %plot( eps,input_cur_p50_means, '-gs' );
    %hold on
    errorbar( eps, input_cur_mean_means,input_cur_mean_stdev, '-gs')%errorbar( eps, input_cur_p50_means,input_cur_p50_stdev, '-gs')%,'MarkerSize',5,'MarkerFaceColor','Green','LineWidth',1.3);
    hold on

    ax_vals = [];
    ax_vals = [ax_vals ; max(input_dn_mean_stdev)];%ax_vals = [ax_vals ; max(input_dn_p50_stdev)];
    ax_vals = [ax_vals ; max(input_dp_mean_stdev)];%ax_vals = [ax_vals ; max(input_dp_p50_stdev)];
    ax_vals = [ax_vals ; max(input_cur_mean_stdev)];%ax_vals = [ax_vals ; max(input_cur_p50_stdev)];

    title(append('Mean Strain Components: ',plot_list, ', Datapoints: ', string(length(dn_mean)) ));%title(append('Median Strain Components: ',plot_list, ', Datapoints: ', string(length(dn_p50)) ));
    if i == 1
        legend('Mean Contraction','Mean Dilation','Mean Shear','Location','northwest') %legend('dn\_p50','dp\_p50','cur\_p50','Location','northwest');
    end
    ylabel('Mean Strain Magnitude');
    xlabel('Normalized Axial Strain');
    xlim([-0.01,eps(end)*1.02]);
    ylim([-4,17]);
    %axis([-0.01 eps(end)*1.02 0 1.05*max(ax_vals)])
    %axis()


    x2 = linspace(1,length(ep2),length(ep2));

    counter = [1];
    for j = 1: length(ep2)-1
        if ep2(j) ~= ep2(j+1)
            counter = [counter ; j];
        end
    end
    counter = [counter ; length(ep2)];

    eps2 = [];
    ep_list2 = [];
    pred_means2 = [];
    pred_stdev2 = [];

    for k = 1: length(counter)-1
        eps2 = [eps2; ep2(counter(k+1))];
        ep_list2 = [ep_list2;counter(k)+round((counter(k+1) - counter(k))/2)];
        pred_means2 = [pred_means2 ; mean( pred2( counter(k):counter(k+1) ) )];
        pred_stdev2 = [pred_stdev2 ; std( pred2( counter(k):counter(k+1) ) )];
    end

    subplot(2,2,2*i)
    p = plot(x2,pred2, x2,ep2);
    set(p,{'LineWidth'},{0.7;1.3})
    p(1).Color = [0.05 0.45 0.98 0.5]; %[0.13 0.43 0.90]; [0.39 0.88 0.15]
    p(2).Color = [0.96 0.5 0.1]; %[0.91 0.41 0.17];
    hold on
    errorbar(ep_list2,pred_means2,pred_stdev2,'-s','MarkerSize',5,'MarkerFaceColor',[1,0,0],'Color',[1,0,0],'LineWidth',1.3); %[0.8 0.2 0.7]
    if rads(i) == 2
    title(append(cname,'Predicted Vs Observed Axial Strain ',plot_list,', R^2 = ',string(round(table2array(M_xgb_20( find( contains(experiment_name,plot_list) ),3)) ,2)) ));
    else
    title(append(cname,'Predicted Vs Observed Axial Strain ',plot_list,', R^2 = ',string(round(table2array(M_xgb_50( find( contains(experiment_name,plot_list) ),3)) ,2)) ));  
    end
    if i == 1
    legend('Model Prediction','Observed Strain','Mean Model Output','Location','northwest');
    end 
    ylabel('Normalized Axial Strain');
    xlabel('Experiment Sample');
    axis([0 length(ep2)+max(x)/100 -0.05 1.05]);
    hold on
    hold off
end

set(gcf,'Position',[100 100 1100 1100])
%set(gcf,'Position',[100 100 1000 2000])
figfile = append(figure_directory,'low_high_resolution_data_model_comparison_',mname,'_figure')
figfile2 = append(figure_directory,'low_high_resolution_data_model_comparison_',mname,'_figure');


saveas(gcf, figfile, 'epsc')
saveas(gcf, figfile2, 'pdf')


figure()

plot_list = ["WG04","MONZ05","FBL02","GRS03"]; %experiment_name = ["FBL01","FBL02","ETNA01","ETNA02","MONZ04","MONZ05","WG01","WG02","WG04","GRS02","GRS03","ANS02","ANS03","ANS04","ANS05"];
color_list = ["-rs","-bs","-gs","-ms"];

for i = 1: length(plot_list)
    resultstring = append('result_',mname,'_',plot_list(i),'_g',string(rad),'0.txt');
    datastring = append('strains_curr_',plot_list(i),'_g',string(rad),'0.txt');
    R = load(append(result_directory,resultstring));
    D = readtable(append(data_directory,datastring));

    ep = R(:,2);
    pred = R(:,1);

    dn_mean = table2array(D(:,11)); %dn_p50 = table2array(D(:,10));
    dp_mean = table2array(D(:,20)); %dp_p50 = table2array(D(:,19));
    cur_mean = table2array(D(:,29)); %cur_p50 = table2array(D(:,28));
    dn_mean = dn_mean(~isnan(dn_mean)); %dn_p50 = dn_p50(~isnan(dn_p50));
    dp_mean = dp_mean(~isnan(dp_mean)); %dp_p50 = dp_p50(~isnan(dp_p50));
    cur_mean = cur_mean(~isnan(cur_mean)); %cur_p50 = cur_p50(~isnan(cur_p50));

    x = linspace(1,length(ep),length(ep));

    counter = [1];
    for j = 1: length(ep)-1
        if ep(j) ~= ep(j+1)
            counter = [counter ; j];
        end
    end
    counter = [counter ; length(ep)];

    eps = [];
    ep_list = [];
    pred_means = [];
    pred_stdev = [];
    input_dn_mean_means = []; %input_dn_p50_means = [];
    input_dp_mean_means = []; %input_dp_p50_means = [];
    input_cur_mean_means = []; %input_cur_p50_means = [];
    input_dn_mean_stdev = []; %input_dn_p50_stdev = [];
    input_dp_mean_stdev = []; %input_dp_p50_stdev = [];
    input_cur_mean_stdev = []; %input_cur_p50_stdev = [];

    for k = 1: length(counter)-1
        eps = [eps; ep(counter(k+1))];
        ep_list = [ep_list;counter(k)+round((counter(k+1) - counter(k))/2)];
        pred_means = [pred_means ; mean( pred( counter(k):counter(k+1) ) )];
        pred_stdev = [pred_stdev ; std( pred( counter(k):counter(k+1) ) )];
        input_dn_mean_means = [input_dn_mean_means ; mean( dn_mean( counter(k):counter(k+1) ) )]; %input_dn_p50_means = [input_dn_p50_means ; mean( dn_p50( counter(k):counter(k+1) ) )];
        input_dp_mean_means = [input_dp_mean_means ; mean( dp_mean( counter(k):counter(k+1) ) )]; %input_dp_p50_means = [input_dp_p50_means ; mean( dp_p50( counter(k):counter(k+1) ) )];
        input_cur_mean_means = [input_cur_mean_means ; mean( cur_mean( counter(k):counter(k+1) ) )]; %input_cur_p50_means = [input_cur_p50_means ; mean( cur_p50( counter(k):counter(k+1) ) )];
        input_dn_mean_stdev = [input_dn_mean_stdev ; std( dn_mean( counter(k):counter(k+1) ) )]; %input_dn_p50_stdev = [input_dn_p50_stdev ; std( dn_p50( counter(k):counter(k+1) ) )];
        input_dp_mean_stdev = [input_dp_mean_stdev ; std( dp_mean( counter(k):counter(k+1) ) )]; %input_dp_p50_stdev = [input_dp_p50_stdev ; std( dp_p50( counter(k):counter(k+1) ) )];
        input_cur_mean_stdev = [input_cur_mean_stdev ; std( cur_mean( counter(k):counter(k+1) ) )]; %input_cur_p50_stdev = [input_cur_p50_stdev ; std( cur_p50( counter(k):counter(k+1) ) )];
    end

    if (i <= 2)
        subplot(3,2,1)
        title('Comparison of Evolution of Mean Contraction');
        xlabel('Normalized Axial Contraction')
        ylabel('Mean Shear')
        errorbar(eps, input_dn_mean_means,input_dn_mean_stdev, color_list(i))%errorbar(eps, input_dn_p50_means,input_dn_p50_stdev, '-rs')%,'MarkerSize',5,'MarkerFaceColor','Red','LineWidth',1.3);
        xlim([-0.01,eps(end)*1.02]);
        legend(plot_list(1:2));
        hold on

        subplot(3,2,3)
        title('Comparison of Evolution of Mean Dilation');
        xlabel('Normalized Axial Strain')
        ylabel('Mean Dilation')
        errorbar( eps, input_dp_mean_means,input_dp_mean_stdev, color_list(i))%errorbar( eps, input_dp_p50_means,input_dp_p50_stdev, '-bs')%,'MarkerSize',5,'MarkerFaceColor','Blue','LineWidth',1.3);
        xlim([-0.01,eps(end)*1.02]);
        hold on

        subplot(3,2,5)
        title('Comparison of Evolution of Mean Shear');
        xlabel('Normalized Axial Strain')
        ylabel('Mean Shear')
        errorbar( eps, input_cur_mean_means,input_cur_mean_stdev, color_list(i))%errorbar( eps, input_cur_p50_means,input_cur_p50_stdev, '-gs')%,'MarkerSize',5,'MarkerFaceColor','Green','LineWidth',1.3);
        xlim([-0.01,eps(end)*1.02]);
        hold on
    else
        subplot(3,2,2)
        title('Comparison of Evolution of Mean Contraction');
        xlabel('Normalized Axial Strain')
        ylabel('Mean Contraction')
        errorbar(eps, input_dn_mean_means,input_dn_mean_stdev, color_list(i))%errorbar(eps, input_dn_p50_means,input_dn_p50_stdev, '-rs')%,'MarkerSize',5,'MarkerFaceColor','Red','LineWidth',1.3);
        xlim([-0.01,eps(end)*1.02]);
        legend(plot_list(3:end));
        hold on

        subplot(3,2,4)
        title('Comparison of Evolution of Mean Dilation');
        xlabel('Normalized Axial Strain')
        ylabel('Mean Dilation')
        errorbar( eps, input_dp_mean_means,input_dp_mean_stdev, color_list(i))%errorbar( eps, input_dp_p50_means,input_dp_p50_stdev, '-bs')%,'MarkerSize',5,'MarkerFaceColor','Blue','LineWidth',1.3);
        xlim([-0.01,eps(end)*1.02]);
        hold on

        subplot(3,2,6)
        title('Comparison of Evolution of Mean Shear');
        xlabel('Normalized Axial Strain')
        ylabel('Mean Shear')
        errorbar( eps, input_cur_mean_means,input_cur_mean_stdev, color_list(i))%errorbar( eps, input_cur_p50_means,input_cur_p50_stdev, '-gs')%,'MarkerSize',5,'MarkerFaceColor','Green','LineWidth',1.3);
        xlim([-0.01,eps(end)*1.02]);
        hold on
    end
end
set(gcf,'Position',[100 100 1200 2000])
%set(gcf,'Position',[100 100 1000 2000])
figfile = append(figure_directory,'input_data_comparison_',mname,'_g',string(rad),'0_figure')
figfile2 = append(figure_directory,'input_data_comparison_',mname,'_g',string(rad),'0_figure');


saveas(gcf, figfile, 'epsc')
saveas(gcf, figfile2, 'pdf') 