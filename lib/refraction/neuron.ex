defmodule Refraction.Neuron do
  alias Refraction.Neuron
  alias Refraction.Neuron.{Activation, Connection}

  @type t :: %__MODULE__{
          pid: pid(),
          input: integer,
          output: integer | float,
          incoming: [integer],
          outgoing: [integer | float],
          bias?: boolean,
          delta: float
        }
  defstruct pid: nil,
            input: 0,
            output: 0,
            incoming: [],
            outgoing: [],
            bias?: false,
            delta: 0

  @spec start_link(map()) :: {:ok, pid()}
  def start_link(attributes \\ %{}) do
    {:ok, pid} = Agent.start_link(fn -> %Neuron{} end)
    update(pid, Map.merge(attributes, %{pid: pid}))
    {:ok, pid}
  end

  @spec get(pid()) :: Neuron.t()
  def get(pid), do: Agent.get(pid, fn neuron -> neuron end)

  @spec learning_rate() :: float
  def learning_rate, do: 0.618

  @spec update(pid(), map()) :: :ok
  def update(pid, new_attributes) do
    Agent.update(pid, fn current_attributes ->
      Map.merge(current_attributes, new_attributes)
    end)
  end

  @spec connect(pid(), pid()) :: :ok
  def connect(source_neuron_pid, target_neuron_pid) do
    {:ok, connection_pid} =
      Connection.connection_for(source_neuron_pid, target_neuron_pid)

    update(source_neuron_pid, %{
      outgoing: get(source_neuron_pid).outgoing ++ [connection_pid]
    })

    update(target_neuron_pid, %{
      incoming: get(target_neuron_pid).incoming ++ [connection_pid]
    })
  end

  @spec activate(pid(), atom, integer | nil) :: :ok
  def activate(neuron_pid, activation, value \\ nil) do
    neuron = get(neuron_pid)

    attributes =
      if neuron.bias? do
        %{output: 1}
      else
        input = value || Enum.reduce(neuron.incoming, 0, calculate_input())
        %{input: input, output: Activation.calculate_output(activation, input)}
      end

    update(neuron_pid, attributes)
  end

  defp calculate_input do
    fn connection_pid, sum ->
      connection = Connection.get(connection_pid)
      sum + get(connection.source_pid).output * connection.weight
    end
  end
end
