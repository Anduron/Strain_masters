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

%h1 = HeatMap(flipud(table2array( T_xgb_50(1:end-4,1:end-4) )));
%h1.Annotate = true;
%addTitle(h1,'Score Matrix')
%addXLabel(h1,'Trained')
%addYLabel(h1,'Tested')
%h1.Xticks(experiment_name(1:end-4))
%hold on

for i = 1: length(experiment_name)


    figure(i)
    resultstring = append('result_',mname,'_',experiment_name(i),'_g',string(rad),'0.txt');
    datastring = append('strains_curr_',experiment_name(i),'_g',string(rad),'0.txt');
    R = load(append(result_directory,resultstring));
    D = readtable(append(data_directory,datastring));
    ep = R(:,2);
    pred = R(:,1);
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
    title(append(cname,'Predicted Vs Observed Axial Strain ',experiment_name(i),', R^2 = ',string(table2array(M_xgb_50(i,3)) ) ));
    legend('Model Prediction','Observed Strain','Mean Model Output','Location','northwest');
    ylabel('Normalized Axial Strain');
    xlabel('Experiment Sample');
    axis([0 length(ep)+100 -0.05 1.05])
    figfile = append(figure_directory,'model_prediction_',mname,'_',experiment_name(i),'_g',string(rad),'0_figure')
    figfile2 = append('figfiles/','model_prediction_',mname,'_',experiment_name(i),'_g',string(rad),'0_figure');
    hold on
    saveas(gcf, figfile, 'epsc')
    saveas(gcf, figfile2, 'fig')
    hold off
end

%short = {'FBL' 'ETNA' 'MONZ' 'WG' 'GRS' 'ANS'};
%figure(length(experiment_name)+1)
%set(figure(length(experiment_name)+1), 'Pos', [488, 342, 864, 420])
%subplot(1,2,1)
%short = {'Sandstone' 'Basalt' 'Monzonite' 'Granite' 'Shale' 'Limestone'};
%b = bar( transpose(scores(1:2,:)) );
%b(1).FaceColor = [0.91 0.41 0.17];
%b(2).FaceColor = [0.13 0.43 0.90];
%title('Test Scores on Low Resolution');
%legend('xgb','svm');
%ylabel('Model R^{2}');
%set(gca,'FontSize',18,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
%hold on

%subplot(1,2,2)
%short = {'Sandstone' 'Basalt' 'Monzonite' 'Shale' 'Granite' 'Limestone'};
%b = bar( transpose(scores(3:4,:)) );
%b(1).FaceColor = [0.91 0.41 0.17];
%b(2).FaceColor = [0.39 0.88 0.15];
%title('Test Score on High Resolution');
%legend('xgb','dnn');
%ylabel('Model R^{2}');
%set(gca,'FontSize',16,'LineWidth',2,'Xtick',[1:6],'XTickLabel',short,'XTickLabelRotation',45)
%saveas(gcf, append(figure_directory,'Bar_Avg_Scores'), 'epsc') 