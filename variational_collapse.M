clc;
format compact;
syms a b c;
C = [1 a b; a 1 c; b c 1];
fprintf(1,'C = \n'); disp(C);
p = charpoly(C);
fprintf(1,'\nCharacteristic polynomial of C:\n'); disp(p);
fprintf(1,'\nC is positive definite if:\n')
fprintf(1,'\t%s > 0\n\t%s > 0\n\t%s > 0\n',-p(2),p(3),-p(4));
invC = inv(C);
fprintf(1,'\nAdjoint(C) = Inverse(C) * det(C):\n'); disp(adjoint(C));
D = diag(invC);
H = (inv(C).^2)./(D*transpose(D));
fprintf(1,'\nH = \n'); disp(H); fprintf(1,'\n');
x = H\[1;1;1];
fprintf(1,'\nHx=1\nx=\n'); disp(x);
fprintf(1,'\nFactors of x * det(H) * (1-a^2)^2*(1-b^2)^2*(1-c^2)^2\n');
for i=1:length(x)
  disp(factor(det(H)*x(i)*(1-a^2)^2*(1-b^2)^2*(1-c^2)^2));
end
J = C.*C;
fprintf(1,'\nJ = \n'); disp(J); fprintf(1,'\n');
x = J\[1;1;1];
fprintf(1,'\nJx=1\nx=\n'); disp(x);
fprintf(1,'\nFactors of x * det(J)\n');
for i=1:length(x)
  disp(factor(det(J)*x(i)));
end
fprintf(1,'\nNo variational collapse if:\n')
fprintf(1,'\ta^2 + b^2 < 1 + c^2\n');
fprintf(1,'\tb^2 + c^2 < 1 + a^2\n');
fprintf(1,'\tc^2 + a^2 < 1 + b^2\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D plots
clear a b;
figure(1); clf;
set(gcf,'Position',[-3300 1329 3072 491]);
blue = [0 0.4470, 0.7410];
red = [0.6350, 0.0780, 0.1840];
for i=1:6
  subplot(1,6,i); axis square;
  c = min(0.2*(i-1),0.99);
  plot_phase_diagram_fast(c);
  % plot_phase_diagram(c);
  set(gca,'XTick',[0 1],'YTick',[0 1],'FontSize',36);
  text(0.5,1.0,sprintf('C_{23} = %g',c),'FontSize',36,... %'FontWeight','bold',...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
  xlabel('C_{12}','Position',[0.5 -0.03 -1]);
  ylabel('C_{13}','Rotation',0,'VerticalAlignment','middle');
  % set(gca,'FontWeight','bold');
end
set(gcf,'Color','w');
subplot(1,6,1);
text(-0.75,0.72,'Variational','VerticalAlignment','bottom',...
    'FontWeight','bold','FontSize',48,'HorizontalAlignment','center');
text(-0.75,0.72,'Collapse','VerticalAlignment','top',...
    'FontWeight','bold','FontSize',48,'HorizontalAlignment','center');
text(-0.75,0.3,' $0<|\Psi|<\infty$ ','Interpreter','LaTeX','EdgeColor','k','LineWidth',2,...
  'VerticalAlignment','bottom','FontSize',40,'Color','w','BackgroundColor',blue,'HorizontalAlignment','center');
text(-0.75,0.2,' $|\Psi|\in\{0,\infty\}$ ','Interpreter','LaTeX','EdgeColor','k','LineWidth',2,...
  'VerticalAlignment','top','FontSize',40,'Color','w','BackgroundColor',red,'HorizontalAlignment','center');
set(gcf,'PaperSize',[44 8],'PaperUnits','inches');
print -dpdf fig_variational_collapse.pdf;
return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_phase_diagram_fast(c)
blue = [0 0.4470, 0.7410];
red = [0.6350, 0.0780, 0.1840];
cla; set(gca,'XLim',[-0.01 1.01],'YLim',[-0.001 1.01],'LineWidth',2,'Box','on');
set(gca,'FontSize',18);
grid on; hold on; 
for a=0:0.005:1
  bl1 = sqrt(max(0,c*c+a*a-1));
  bl2 = a*c-sqrt(max(0,1+a*a*c*c-a*a-c*c));
  bu1 = sqrt(1-abs(a^2-c^2));
  bu2 = a*c+sqrt(max(0,1+a*a*c*c-a*a-c*c));
  plot([a a],[max(bl1,bl2),min(bu1,bu2)],'-','Color',blue,'LineWidth',1.5);
  if (a>0)
    plot([a a],[min(bu1,bu2),max(bu1,bu2)],'-','Color',red,'LineWidth',1.5);
  end
  if (max(bl1,bl2)>0)
    plot([a a],[min(bl1,bl2),max(bl1,bl2)],'-','Color',red,'LineWidth',1.5);
  end
end
drawnow;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_phase_diagram(c)
blue = [0 0.4470, 0.7410];
red = [0.6350, 0.0780, 0.1840];
cla; set(gca,'XLim',[0 1],'YLim',[0 1],'LineWidth',2);
set(gca,'FontSize',18);
grid on; hold on; 
drawnow;
inc=0.04;
for a=inc:inc:1-inc
  for b=inc:inc:1-inc
    C = [1 a b; a 1 c; b c 1];
    if (any(eig(C)<=eps))
      continue;
    end
    if (rcond(C)<eps)
      keyboard;
    end
    invC = inv(C);
    D = diag(invC);
    H = (invC.^2)./(D*D');
    x = H\[1;1;1];
    if (any(x<0))
      plot(a,b,'o','MarkerSize',4,'MarkerEdgeColor',red,'MarkerFaceColor',red);
    else
      plot(a,b,'o','MarkerSize',4,'MarkerEdgeColor',blue,'MarkerFaceColor',blue);
    end
    y = (C.*C)\[1;1;1];
    if (any(x<0)~=any(y<0))
      fprintf(1,'(a,b,c)=(%f,%f,%f),x=(%f,%f,%f),y=(%f,%f,%f)\n',...
        a,b,c,x(1),x(2),x(3),y(1),y(2),y(3));
    end
  end
  drawnow;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
